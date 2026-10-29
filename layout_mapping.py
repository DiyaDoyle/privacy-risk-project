from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from paddleocr import LayoutDetection


# Resolve paths relative to this script, not the terminal's working directory.
ROOT = Path(__file__).resolve().parent
PHASE2_ROOT = ROOT / "evaluation" / "phase2_results"
STAGE1_RESULTS_DIR = PHASE2_ROOT / "stage1"
IMAGES_DIR = ROOT / "evaluation" / "dataset" / "images"
OUTPUT_DIR = PHASE2_ROOT / "stage2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYOUT_MODEL_NAME = "PP-DocLayout-L"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def valid_box(box: Any) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, box)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_inside(point: tuple[float, float], box: list[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def calculate_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def extract_layout(result: Any) -> list[dict[str, Any]]:
    raw = getattr(result, "json", None)
    if callable(raw):
        raw = raw()
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        return []

    payload = raw.get("res", raw)
    if not isinstance(payload, dict):
        return []

    regions = []
    for index, item in enumerate(payload.get("boxes", [])):
        if not isinstance(item, dict):
            continue
        box = valid_box(item.get("coordinate", item.get("bbox")))
        if box is None:
            continue
        try:
            confidence = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        regions.append({
            "region_id": index,
            "label": str(item.get("label", "unknown")),
            "score": confidence,
            "bbox": box,
        })
    return regions


def find_layout_region(pii_box: list[float] | None,
                       layout_regions: list[dict[str, Any]]):
    if pii_box is None:
        return None, "unmatched_missing_pii_bbox"

    center = box_center(pii_box)
    containing = [r for r in layout_regions if point_inside(center, r["bbox"])]
    if containing:
        containing.sort(key=lambda r: (
            (r["bbox"][2] - r["bbox"][0]) *
            (r["bbox"][3] - r["bbox"][1])
        ))
        return containing[0], "center_inside_smallest_region"

    best_region = None
    best_iou = 0.0
    for region in layout_regions:
        overlap = calculate_iou(pii_box, region["bbox"])
        if overlap > best_iou:
            best_iou = overlap
            best_region = region
    if best_region is not None and best_iou > 0:
        return best_region, "highest_iou"
    return None, "unmatched_no_overlap"


def get_font(size: int = 12):
    for candidate in (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                pass
    return ImageFont.load_default()


def annotate_image(image_path: Path, regions: list[dict[str, Any]],
                   mappings: list[dict[str, Any]], output_path: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    label_font = get_font()

    for region in regions:
        box = [int(round(v)) for v in region["bbox"]]
        draw.rectangle(box, outline="blue", width=2)
        x1, y1, _, _ = box
        draw.text((x1, max(0, y1 - 13)),
                  f'{region["region_id"]}: {region["label"]}',
                  fill="blue", font=label_font)

    for item in mappings:
        box = item.get("pii_bbox")
        if not box:
            continue
        draw.rectangle([int(round(v)) for v in box],
                       outline="red" if item.get("layout_region_id") is not None else "orange",
                       width=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def process_document(stage1_path: Path, image_path: Path, model: Any) -> dict[str, Any]:
    stage1_data = load_json(stage1_path)
    if not isinstance(stage1_data, dict):
        raise ValueError(f"Expected Stage 1 JSON object in {stage1_path}")

    # This is the actual model output from phase2_run_models.py.
    # Ground-truth comparison records are intentionally not used here.
    predictions = stage1_data.get("mapped_predictions")
    if not isinstance(predictions, list):
        raise ValueError(
            f"{stage1_path} has no 'mapped_predictions' list. "
            "Run phase2_run_models.py first; ground_truth_comparisons are evaluation-only."
        )

    with Image.open(image_path) as image:
        width, height = image.size

    result_list = list(model.predict(input=str(image_path), batch_size=1))
    if not result_list:
        raise RuntimeError("PP-DocLayout returned no result.")
    regions = extract_layout(result_list[0])

    mappings = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        box = valid_box(prediction.get("predicted_bbox"))
        region, method = find_layout_region(box, regions)
        mappings.append({
            "pii_text": str(prediction.get("text", "")),
            "pii_type": str(prediction.get("type", "")),
            "confidence": prediction.get("score"),
            "pii_bbox": box,
            "ocr_words": prediction.get("ocr_words", []),
            "layout_region_id": region["region_id"] if region else None,
            "layout_region_type": region["label"] if region else None,
            "layout_region_score": region["score"] if region else None,
            "layout_region_bbox": region["bbox"] if region else None,
            "mapping_method": method,
        })

    mapped_count = sum(item["layout_region_id"] is not None for item in mappings)
    document = stage1_data.get("document", stage1_path.stem.removesuffix("_stage1"))
    output_json = OUTPUT_DIR / f"{document}_layout.json"
    output_image = OUTPUT_DIR / f"{document}_layout.png"

    save_json(output_json, {
        "document": document,
        "image_width": width,
        "image_height": height,
        "layout_model": LAYOUT_MODEL_NAME,
        "layout_region_count": len(regions),
        "pii_prediction_count": len(mappings),
        "pii_layout_mapped_count": mapped_count,
        "pii_layout_unmapped_count": len(mappings) - mapped_count,
        "layout_mapping_coverage": mapped_count / len(mappings) if mappings else None,
        "layout_regions": regions,
        "pii_mappings": mappings,
    })
    annotate_image(image_path, regions, mappings, output_image)

    print(f"{document}: predictions={len(mappings)}, mapped={mapped_count}, regions={len(regions)}")
    return {
        "document": document,
        "predictions": len(mappings),
        "mapped": mapped_count,
        "regions": len(regions),
        "output_json": str(output_json),
        "output_image": str(output_image),
    }


def main() -> None:
    print("=" * 70)
    print("STAGE 2 — PREDICTED PII TO LAYOUT REGION MAPPING")
    print("=" * 70)
    print(f"Stage 1 results: {STAGE1_RESULTS_DIR}")
    print(f"Input images:    {IMAGES_DIR}")
    print(f"Output:          {OUTPUT_DIR}")

    if not STAGE1_RESULTS_DIR.is_dir():
        raise SystemExit(
            f"Stage 1 results directory not found: {STAGE1_RESULTS_DIR}\n"
            "Run phase2_run_models.py first."
        )

    stage1_files = sorted(STAGE1_RESULTS_DIR.glob("*_stage1.json"))
    if not stage1_files:
        raise SystemExit(
            f"No *_stage1.json files found in {STAGE1_RESULTS_DIR}. "
            "Run phase2_run_models.py first."
        )

    print("Loading PP-DocLayout-L...")
    model = LayoutDetection(
        model_name=LAYOUT_MODEL_NAME,
        device="cpu",
        enable_mkldnn=False,
    )
    print("Model loaded.")

    successes = []
    failures = []
    for stage1_path in stage1_files:
        document = stage1_path.name.removesuffix("_stage1.json")
        image_path = IMAGES_DIR / f"{document}.png"
        if not image_path.is_file():
            failures.append({"document": document, "error": f"Image not found: {image_path}"})
            print(f"SKIP {document}: image not found: {image_path}")
            continue
        try:
            successes.append(process_document(stage1_path, image_path, model))
        except Exception as exc:
            failures.append({"document": document, "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR {document}: {type(exc).__name__}: {exc}")

    summary = {
        "layout_model": LAYOUT_MODEL_NAME,
        "documents_processed": len(successes),
        "documents_failed": len(failures),
        "documents": successes,
        "errors": failures,
    }
    save_json(OUTPUT_DIR / "stage2_summary.json", summary)
    print(f"\nFinished: {len(successes)} processed, {len(failures)} failed.")
    print(f"Results: {OUTPUT_DIR}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
