
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

import phase2_run_models as models
import lcprs_v2 as lcprs


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "lcprs_config.json"
OUTPUT_ROOT = ROOT / "evaluation" / "new_document_results"

SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_INPUTS = SUPPORTED_IMAGES | {".pdf"}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def load_project_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as f:
        config = json.load(f)

    masking = config.get("masking", {})
    if not masking.get("enabled", True):
        raise ValueError(
            "Masking is disabled. This entry point requires masking.enabled=true."
        )

    if not masking.get("mask_all_detected_pii", True):
        raise ValueError(
            "Only mask_all_detected_pii=true is implemented in this version."
        )

    color = masking.get("fill_color", [0, 0, 0])
    if (
        not isinstance(color, list)
        or len(color) != 3
        or any(not isinstance(v, int) or not 0 <= v <= 255 for v in color)
    ):
        raise ValueError("masking.fill_color must be three integers from 0 to 255.")

    padding = masking.get("bbox_padding_pixels", 2)
    if not isinstance(padding, int) or padding < 0:
        raise ValueError("masking.bbox_padding_pixels must be a non-negative integer.")

    return config


def safe_name(path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return name or "document"


def load_pages(input_path: Path, temp_dir: Path):
    """
    Return (page_images, page_temp_paths, source_kind).

    PDF pages are rendered to PNG at a fixed 2x scale.
    Image inputs are copied to RGB without changing their dimensions.
    """
    page_images = []
    page_paths = []

    if input_path.suffix.lower() == ".pdf":
        pdf = pymupdf.open(str(input_path))
        try:
            if pdf.needs_pass:
                raise ValueError(
                    "This PDF is password-protected. Decrypt it with authorization first."
                )

            if len(pdf) == 0:
                raise ValueError("The PDF contains no pages.")

            for index, page in enumerate(pdf):
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False,
                )
                image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                page_path = temp_dir / f"page_{index + 1:04d}.png"
                image.save(page_path, format="PNG")

                page_images.append(image)
                page_paths.append(page_path)
        finally:
            pdf.close()

        return page_images, page_paths, "pdf"

    if input_path.suffix.lower() not in SUPPORTED_IMAGES:
        raise ValueError(f"Unsupported input extension: {input_path.suffix}")

    with Image.open(input_path) as source:
        image = source.convert("RGB")

    page_path = temp_dir / "page_0001.png"
    image.save(page_path, format="PNG")

    return [image], [page_path], "image"


def detect_pii_and_layout(image_path: Path, ocr, gliner, layout_model, labels):
    """
    Use the same detector models and matching helpers as the existing
    phase2_run_models.py evaluation pipeline. Ground-truth annotations
    are not used.
    """
    ocr_results = list(ocr.predict(input=str(image_path)))
    if not ocr_results:
        raise RuntimeError("PaddleOCR returned no result.")

    text, words = models.ocr_words(ocr_results[0])
    if not text.strip():
        raise RuntimeError(
            "OCR extracted no text. The page cannot be declared safe."
        )

    coordinate_words = [w for w in words if w.get("bbox") is not None]

    result = gliner.extract_entities(
        text,
        labels,
        threshold=models.GLINER_THRESHOLD,
        include_confidence=True,
        include_spans=True,
    )
    detected = models.flatten_gliner(result)

    mappings = []
    entities_without_bbox = []

    for entity in detected:
        matched_words = models.find_words(entity["text"], coordinate_words)
        bbox = models.union_box(matched_words)

        if bbox is None:
            # Preserve the failure in the audit. Never silently treat
            # an entity without coordinates as masked.
            entities_without_bbox.append({
                "pii_type": entity["type"],
                "confidence": entity["score"],
                "reason": "PII entity could not be matched to OCR coordinates",
            })
            continue

        mappings.append({
            "pii_text": entity["text"],
            "pii_type": entity["type"],
            "confidence": entity["score"],
            "pii_bbox": bbox,
            "ocr_words": [w["text"] for w in matched_words],
        })

    layout_results = list(
        layout_model.predict(input=str(image_path), batch_size=1)
    )
    if not layout_results:
        raise RuntimeError("PP-DocLayout-L returned no result.")

    regions = models.extract_layout(layout_results[0])

    for item in mappings:
        region, method = models.map_to_layout(item["pii_bbox"], regions)
        item["layout_region_id"] = region["region_id"] if region else None
        item["layout_region_type"] = region["label"] if region else None
        item["layout_region_score"] = region["score"] if region else None
        item["layout_region_bbox"] = region["bbox"] if region else None
        item["mapping_method"] = method

    return mappings, regions, entities_without_bbox, len(detected), len(words)


def mask_image(image: Image.Image, mappings: list, masking_config: dict):
    """
    Draw opaque rectangles over mapped PII on a copy of the page.

    Fail closed if any detected entity lacks a usable bounding box.
    """
    if not masking_config.get("mask_all_detected_pii", True):
        raise ValueError("Selective masking is not supported.")

    padding = masking_config.get("bbox_padding_pixels", 2)
    color = tuple(masking_config.get("fill_color", [0, 0, 0]))
    fail_on_missing = masking_config.get("fail_on_missing_bbox", True)

    width, height = image.size
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)

    masked_types = Counter()
    invalid = []

    for index, item in enumerate(mappings):
        bbox = item.get("pii_bbox")

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            invalid.append(index)
            continue

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            invalid.append(index)
            continue

        if not all(map(lambda v: v == v and abs(v) != float("inf"),
                       [x1, y1, x2, y2])):
            invalid.append(index)
            continue

        if x2 <= x1 or y2 <= y1:
            invalid.append(index)
            continue

        # Clamp to page bounds and add configurable padding.
        left = max(0, min(width, int(x1) - padding))
        top = max(0, min(height, int(y1) - padding))
        right = max(0, min(width, int(x2 + padding + 0.999)))
        bottom = max(0, min(height, int(y2 + padding + 0.999)))

        if right <= left or bottom <= top:
            invalid.append(index)
            continue

        draw.rectangle([left, top, right, bottom], fill=color)
        masked_types[str(item.get("pii_type", "unknown"))] += 1

    if invalid and fail_on_missing:
        raise RuntimeError(
            f"{len(invalid)} mapped PII entities could not be masked. "
            "No output will be reported as complete."
        )

    return output, {
        "detector_entities_mapped": len(mappings),
        "entities_masked": sum(masked_types.values()),
        "entities_with_invalid_bbox": len(invalid),
        "masked_types": dict(masked_types),
        "masking_complete_for_mapped_entities": not invalid,
        "method": "opaque rectangles on rasterized page",
    }


def save_flattened_pdf(images: list[Image.Image], output_path: Path):
    """
    Build a new image-only PDF from masked pages.
    Original PDF text objects are not copied into this output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pymupdf.open()

    try:
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            page = result.new_page(width=image.width, height=image.height)
            page.insert_image(page.rect, stream=buffer.getvalue())

        result.save(str(output_path), garbage=4, deflate=True)
    finally:
        result.close()


def process_new_document(input_path: Path, ocr, gliner, layout_model, config):
    input_path = input_path.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() not in SUPPORTED_INPUTS:
        raise ValueError(
            f"Supported formats: {', '.join(sorted(SUPPORTED_INPUTS))}"
        )

    document_name = safe_name(input_path)
    output_dir = OUTPUT_ROOT / document_name
    temp_dir = output_dir / "_work"
    stage2_dir = output_dir / "stage2"
    masked_pages_dir = output_dir / "masked_pages"

    temp_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)
    masked_pages_dir.mkdir(parents=True, exist_ok=True)

    images, page_paths, source_kind = load_pages(input_path, temp_dir)
    page_reports = []
    masked_images = []
    all_audit = []

    print(f"\nInput: {input_path}")
    print(f"Pages: {len(images)}")
    print("Processing does not use ground-truth annotation files.")

    for index, (image, page_path) in enumerate(zip(images, page_paths), start=1):
        print(f"\nPage {index}/{len(images)}: {page_path.name}")

        mappings, regions, unmatched, detected_count, ocr_word_count = (
            detect_pii_and_layout(
                page_path, ocr, gliner, layout_model, config["labels"]
            )
        )

        # Fail closed when a detected PII entity cannot be mapped to page pixels.
        # Producing a masked file in this case could falsely imply complete masking.
        if unmatched and config.get("masking", {}).get("fail_on_missing_bbox", True):
            entity_types = sorted({str(item.get("pii_type", "unknown")) for item in unmatched})
            raise RuntimeError(
                f"Page {index}: {len(unmatched)} detected PII entity/ies could not be "
                f"mapped to OCR boxes (types: {', '.join(entity_types)}). "
                "Masking aborted because masking.fail_on_missing_bbox=true. "
                "No masked output should be treated as complete."
            )

        # LCPRS v3 function and config are imported from lcprs_v2.py.
        risk = lcprs.calculate_lcprs(mappings, image.width, image.height)

        stage2_data = {
            "document": document_name,
            "page": index,
            "image_width": image.width,
            "image_height": image.height,
            "layout_region_count": len(regions),
            "pii_prediction_count": len(mappings),
            "pii_layout_mapped_count": sum(
                item.get("layout_region_id") is not None for item in mappings
            ),
            "layout_regions": regions,
            "pii_mappings": mappings,
        }
        save_json(stage2_dir / f"page_{index:04d}_layout.json", stage2_data)

        masked, masking_audit = mask_image(
            image, mappings, config.get("masking", {})
        )
        masked_images.append(masked)

        masked_page_path = masked_pages_dir / f"page_{index:04d}_masked.png"
        masked.save(masked_page_path, format="PNG")

        # Avoid copying actual detected PII text into the masking audit.
        audit = {
            "page": index,
            "source": input_path.name,
            "detector_entities_found": detected_count,
            "entities_mapped_to_boxes": len(mappings),
            "entities_without_boxes": unmatched,
            "ocr_word_count": ocr_word_count,
            **masking_audit,
        }
        save_json(output_dir / f"page_{index:04d}_masking_audit.json", audit)
        all_audit.append(audit)

        risk_report = {
            "lcprs_version": "3.0-proposed-layout-pairwise",
            "document": document_name,
            "page": index,
            "image_width": image.width,
            "image_height": image.height,
            "risk": risk,
            "detector_entities_found": detected_count,
            "entities_mapped_to_boxes": len(mappings),
            "entities_without_boxes": len(unmatched),
            "stage2_file": str(
                (stage2_dir / f"page_{index:04d}_layout.json").relative_to(ROOT)
            ),
            "masked_page": str(masked_page_path.relative_to(ROOT)),
        }
        save_json(output_dir / f"page_{index:04d}_lcprs_report.json", risk_report)

        page_reports.append({
            "page": index,
            "score": risk.get("score"),
            "risk_category": risk.get("risk_category"),
            "pii_entities_detected": detected_count,
            "pii_entities_mapped": len(mappings),
            "pii_entities_without_boxes": len(unmatched),
            "masked_entities": masking_audit["entities_masked"],
            "masking_complete_for_mapped_entities":
                masking_audit["masking_complete_for_mapped_entities"],
        })

        print(f"  Detected PII entities: {detected_count}")
        print(f"  Entities with boxes:   {len(mappings)}")
        print(f"  Entities without box:  {len(unmatched)}")
        print(f"  LCPRS: {risk.get('score')} ({risk.get('risk_category')})")
        print(f"  Masked mapped entities: {masking_audit['entities_masked']}")

    # Save final output.
    if source_kind == "pdf":
        masked_output = output_dir / f"{document_name}_masked.pdf"
        save_flattened_pdf(masked_images, masked_output)
    else:
        masked_output = output_dir / f"{document_name}_masked.png"
        masked_images[0].save(masked_output, format="PNG")

    # Do not invent a new document-level score by summing page scores.
    # Report each page's score and the maximum page score as a screening
    # summary, explicitly labelled as such.
    page_scores = [
        row["score"] for row in page_reports
        if isinstance(row["score"], (int, float))
    ]
    summary = {
        "document": document_name,
        "input_file": input_path.name,
        "source_type": source_kind,
        "pages_processed": len(page_reports),
        "page_results": page_reports,
        "maximum_page_lcprs_score": max(page_scores) if page_scores else None,
        "maximum_page_score_note": (
            "Screening summary only; this is not a separately validated "
            "document-level LCPRS formula."
        ),
        "masked_output": str(masked_output.relative_to(ROOT)),
        "masking_audit": all_audit,
        "research_status": (
            "LCPRS pairwise layout model is proposed and requires empirical "
            "validation. Masking protects detected entities only; detection "
            "recall must be evaluated separately."
        ),
    }
    save_json(output_dir / "document_report.json", summary)

    # Remove intermediate rendered source pages; retain stage2 results,
    # masked output, per-page reports, and audit.
    for page_path in page_paths:
        try:
            page_path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        temp_dir.rmdir()
    except OSError:
        pass

    print("\n" + "=" * 68)
    print("NEW DOCUMENT PROCESSING FINISHED")
    print("=" * 68)
    print(f"Masked output: {masked_output}")
    print(f"Document report: {output_dir / 'document_report.json'}")
    print(f"Stage 2 results: {stage2_dir}")
    print("Review every page before relying on the masked output.")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run layout-aware PII analysis, LCPRS v3, and masking."
    )
    parser.add_argument(
        "input",
        help="Path to a PDF, PNG, JPG, JPEG, TIFF, or BMP document.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    config = load_project_config()

    # Validate LCPRS v3 configuration before loading the models.
    lcprs.validate_configuration()

    print("Loading PaddleOCR...")
    ocr = models.PaddleOCR(
        lang="en",
        enable_mkldnn=False,
        return_word_box=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    print("Loading GLiNER2-PII...")
    gliner = models.GLiNER2.from_pretrained(models.GLINER_MODEL)

    print("Loading PP-DocLayout-L...")
    layout_model = models.LayoutDetection(
        model_name=models.LAYOUT_MODEL,
        device="cpu",
        enable_mkldnn=False,
    )

    process_new_document(
        input_path,
        ocr,
        gliner,
        layout_model,
        {
            "labels": models.load_configuration()["labels"],
            "masking": config.get("masking", {}),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPROCESSING FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
