import io
import re
import tempfile
import textwrap
from pathlib import Path

import streamlit as st
from PIL import Image
import pymupdf
import cv2
import numpy as np

from paddleocr import PaddleOCR, LayoutDetection
from gliner2 import GLiNER2

# Reuse the EXISTING Phase 2 implementation.
# Nothing in phase2_run_models.py is modified by this temporary app.
from phase2_run_models import (
    load_configuration,
    ocr_words,
    flatten_gliner,
    find_words,
    union_box,
    extract_layout,
    map_to_layout,
    run_lcprs,
    final_image,
    GLINER_MODEL,
    GLINER_THRESHOLD,
    LAYOUT_MODEL,
)


# Render Markdown/HTML without Python indentation turning HTML into a code block.
def md(body, **kwargs):
    return st.markdown(textwrap.dedent(body), **kwargs)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Layout-Contextual Privacy Risk Assessment",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# UI STYLE
# ============================================================

md(
    """
<style>
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 2.5rem;
}

.hero { padding: 4px 0 28px 0; }

.eyebrow {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #8fa3bd;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 24px;
    margin-bottom: 12px;
}

.metric-card {
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 18px;
    background: #182230;
    min-height: 92px;
}

.metric-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    color: #8fa3bd;
    text-transform: uppercase;
}

.metric-value {
    font-size: 27px;
    font-weight: 750;
    color: #f8fafc;
    margin-top: 7px;
}

.risk-box {
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 17px 19px;
    background: #182230;
    margin: 15px 0;
    color: #e5e7eb;
}

.risk-box-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    color: #8fa3bd;
    text-transform: uppercase;
    margin-bottom: 7px;
}

.risk-box-text {
    font-size: 15px;
    color: #e5e7eb;
    line-height: 1.5;
}

.file-card {
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 11px 14px;
    background: #182230;
    color: #e5e7eb;
    margin: 12px 0;
}

.footer-note {
    text-align: center;
    color: #64748b;
    font-size: 11px;
    margin-top: 42px;
    padding-top: 18px;
    border-top: 1px solid #293548;
}

div.stButton > button[kind="primary"] {
    min-height: 46px;
    border-radius: 9px;
    font-weight: 700;
}

div.stButton > button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="eyebrow">PRIVACY ANALYTICS</div>',
    unsafe_allow_html=True,
)

st.title("Layout-Contextual Privacy Risk Assessment")

st.markdown(
    "Analyze document content, identify sensitive information, "
    "understand its spatial context, and calculate privacy risk."
)


# ============================================================
# UPLOAD
# ============================================================

md(
    '<div class="section-title">Document Analysis</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a document",
    type=[
        "pdf",
        # Common raster image formats
        "png",
        "jpg",
        "jpeg",
        "webp",
        "bmp",
        "tif",
        "tiff",
        "gif",
        "ppm",
        "pgm",
        "pbm",
        "pnm",
        "ico",
        "jp2",
        "j2k",
        "jpf",
        "jpx",
        "jpm",
        "mj2",
        "avif",
    ],
    label_visibility="visible",
    help="Supported: PDF, PNG, JPG/JPEG, WEBP, BMP, TIFF, GIF, PPM/PGM/PBM/PNM, ICO, JPEG 2000 (JP2/J2K/JPF/JPX/JPM/MJ2), and AVIF.",
)
if uploaded_file is None:
    st.info("Select a document to begin analysis.")
    st.stop()


# Read the uploaded file ONCE.
file_bytes = uploaded_file.getvalue()
file_size_mb = len(file_bytes) / (1024 * 1024)

md(
    f"""
<div class="file-card">
    <b>{uploaded_file.name}</b>
    <span style="color:#9ca3af;">
        &nbsp; · &nbsp; {file_size_mb:.2f} MB
    </span>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DOCUMENT -> PAGE IMAGES
# ============================================================

def load_pages(data, filename):
    """Convert an uploaded PDF/image into PIL RGB page images."""

    if filename.lower().endswith(".pdf"):
        pdf = pymupdf.open(
            stream=data,
            filetype="pdf",
        )

        pages = []

        try:
            for page in pdf:
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False,
                )

                image = Image.frombytes(
                    "RGB",
                    [pix.width, pix.height],
                    pix.samples,
                )

                pages.append(image)
        finally:
            pdf.close()

        return pages

    image = Image.open(
        io.BytesIO(data)
    ).convert("RGB")

    return [image]


# ============================================================
# CACHED MODEL LOADERS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_config():
    return load_configuration()


@st.cache_resource(show_spinner=False)
def load_ocr():
    return PaddleOCR(
        lang="en",
        enable_mkldnn=False,
        return_word_box=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


@st.cache_resource(show_spinner=False)
def load_gliner():
    return GLiNER2.from_pretrained(
        GLINER_MODEL
    )


@st.cache_resource(show_spinner=False)
def load_layout():
    return LayoutDetection(
        model_name=LAYOUT_MODEL,
        device="cpu",
        enable_mkldnn=False,
    )


# ============================================================
# DOCUMENT-LEVEL RECOVERY / VISUAL ELEMENTS
# ============================================================

def simple_norm(value):
    return re.sub(r"[^a-zA-Z0-9]", "", str(value)).lower()


def word_union(words):
    if not words:
        return None
    return [
        min(float(w["bbox"][0]) for w in words),
        min(float(w["bbox"][1]) for w in words),
        max(float(w["bbox"][2]) for w in words),
        max(float(w["bbox"][3]) for w in words),
    ]


def recover_name_from_id_fields(words, predictions):
    """Recover a missing first-name token only when an explicit ID name field exists.

    This is a conservative post-processing step for structured identity documents.
    It does not hardcode the user's name and does not change the Phase 2 experiment.
    """
    if not words:
        return predictions

    normalized = [(w, simple_norm(w.get("text", ""))) for w in words]
    first_label_indices = [i for i, (_, t) in enumerate(normalized) if t in {"firstname", "givenname"}]
    if not first_label_indices:
        # Also support separate OCR words: FIRST + NAME.
        for i in range(len(normalized) - 1):
            if normalized[i][1] == "first" and normalized[i + 1][1] == "name":
                first_label_indices.append(i)

    if not first_label_indices:
        return predictions

    # Existing person detections, if any.
    person_types = {"person", "person_name", "name", "full_name"}
    existing_person = [p for p in predictions if simple_norm(p.get("type", "")) in person_types]

    for label_i in first_label_indices:
        label_word = normalized[label_i][0]
        lx1, ly1, lx2, ly2 = map(float, label_word["bbox"])
        label_cx = (lx1 + lx2) / 2
        label_bottom = ly2
        label_h = max(1.0, ly2 - ly1)

        # Candidate value immediately below/near the FIRST NAME label.
        candidates = []
        for w, t in normalized:
            if not t or t in {"first", "name", "firstname", "givenname", "last", "lastname"}:
                continue
            x1, y1, x2, y2 = map(float, w["bbox"])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if y1 >= label_bottom - 0.5 * label_h and y1 <= label_bottom + 6 * label_h:
                if abs(cx - label_cx) <= 10 * label_h:
                    candidates.append((abs(cy - label_bottom) + abs(cx - label_cx) * 0.15, w))
        candidates.sort(key=lambda x: x[0])
        if not candidates:
            continue

        first_word = candidates[0][1]
        first_text = str(first_word.get("text", "")).strip()
        if len(simple_norm(first_text)) < 2:
            continue

        # If an existing person detection is nearby, prepend the recovered first name.
        best = None
        best_dist = float("inf")
        fx1, fy1, fx2, fy2 = map(float, first_word["bbox"])
        for p in existing_person:
            bx1, by1, bx2, by2 = map(float, p["predicted_bbox"])
            # Horizontal/vertical proximity check.
            dist = abs(((fx1 + fx2) / 2) - ((bx1 + bx2) / 2)) + abs(((fy1 + fy2) / 2) - ((by1 + by2) / 2))
            if dist < best_dist and dist < max(fx2 - fx1, fy2 - fy1, bx2 - bx1, by2 - by1) * 12:
                best = p
                best_dist = dist

        if best is not None:
            old = str(best.get("text", "")).strip()
            if first_text.lower() not in old.lower().split():
                best["text"] = f"{first_text} {old}"
                best["predicted_bbox"] = word_union([first_word, {"bbox": best["predicted_bbox"]}])
                best["ocr_words"] = [first_text] + list(best.get("ocr_words", []))
            return predictions

        # No person detection existed: create a conservative person prediction from the field.
        predictions.append({
            "text": first_text,
            "type": "person",
            "score": 0.90,
            "predicted_bbox": list(first_word["bbox"]),
            "ocr_words": [first_text],
        })
        return predictions

    return predictions


def find_pan_box(words):
    """Find a PAN-shaped government-ID number without exposing its value in the UI."""
    # OCR can split PAN into multiple tokens, so test each line and a joined version.
    by_line = {}
    for w in words:
        by_line.setdefault(w.get("line", 0), []).append(w)

    pattern = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")
    for line_words in by_line.values():
        line_words = sorted(line_words, key=lambda w: w["bbox"][0])
        for start in range(len(line_words)):
            joined = ""
            selected = []
            for end in range(start, min(len(line_words), start + 8)):
                token = re.sub(r"[^A-Za-z0-9]", "", str(line_words[end].get("text", ""))).upper()
                if not token:
                    continue
                joined += token
                selected.append(line_words[end])
                if pattern.fullmatch(joined):
                    return word_union(selected)
                if len(joined) > 10:
                    break
    return None


def find_qr_codes(image):
    """Detect visible QR codes without exposing decoded contents."""
    cv_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    detector = cv2.QRCodeDetector()
    boxes = []

    try:
        result = detector.detectAndDecodeMulti(cv_image)
        if len(result) == 4:
            ok, decoded_info, points, _ = result
            if ok and points is not None:
                for pts in points:
                    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
                    if len(pts) >= 4:
                        boxes.append([
                            float(pts[:, 0].min()),
                            float(pts[:, 1].min()),
                            float(pts[:, 0].max()),
                            float(pts[:, 1].max()),
                        ])
    except Exception:
        pass

    if not boxes:
        try:
            decoded, points, _ = detector.detectAndDecode(cv_image)
            if points is not None:
                pts = np.asarray(points, dtype=float).reshape(-1, 2)
                if len(pts) >= 4:
                    boxes.append([
                        float(pts[:, 0].min()),
                        float(pts[:, 1].min()),
                        float(pts[:, 0].max()),
                        float(pts[:, 1].max()),
                    ])
        except Exception:
            pass

    return boxes


def find_signature_region(words, regions):
    """Find a signature area using layout labels or an explicit OCR signature label."""
    signature_regions = [
        r for r in regions
        if any(k in simple_norm(r.get("label", "")) for k in ("signature", "sign"))
    ]
    if signature_regions:
        return signature_regions[0]["bbox"]

    sig_words = [w for w in words if "signature" in simple_norm(w.get("text", "")) or simple_norm(w.get("text", "")) == "sign"]
    if not sig_words:
        return None

    # Prefer a layout region containing the signature label.
    for w in sig_words:
        cx = (float(w["bbox"][0]) + float(w["bbox"][2])) / 2
        cy = (float(w["bbox"][1]) + float(w["bbox"][3])) / 2
        containing = []
        for r in regions:
            x1, y1, x2, y2 = r["bbox"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                containing.append(r)
        if containing:
            containing.sort(key=lambda r: (r["bbox"][2]-r["bbox"][0])*(r["bbox"][3]-r["bbox"][1]))
            return containing[0]["bbox"]
    return word_union(sig_words)


def annotate_document_elements(image, elements):
    """Overlay non-PII document elements on the already-generated final image."""
    from PIL import ImageDraw, ImageFont
    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for element in elements:
        box = element.get("bbox")
        label = element.get("label", "")
        if not box:
            continue
        x1, y1, x2, y2 = map(int, box)
        # Keep these visually distinct from the existing PII red boxes.
        draw.rectangle((x1, y1, x2, y2), outline=(255, 165, 0), width=3)
        draw.text((x1, max(0, y1 - 16)), label, fill=(255, 165, 0), font=font)
    return out


# ============================================================
# ANALYZE
# ============================================================

if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False

file_key = f"{uploaded_file.name}:{len(file_bytes)}"
if st.session_state.get("file_key") != file_key:
    st.session_state.file_key = file_key
    st.session_state.analysis_running = False

analyze_clicked = st.button(
    "Analyze Document",
    type="primary",
    width="stretch",
    disabled=st.session_state.analysis_running,
)

if analyze_clicked:
    st.session_state.analysis_running = True

    # --------------------------------------------------------
    # READ DOCUMENT
    # --------------------------------------------------------

    try:
        pages = load_pages(
            file_bytes,
            uploaded_file.name,
        )

        if not pages:
            raise RuntimeError(
                "The document contains no readable pages."
            )

    except Exception as error:
        st.error(
            f"Unable to read document: {error}"
        )
        st.stop()

    st.success(
        f"Document ready: {len(pages)} page(s)"
    )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    status = st.status(
        "Preparing analysis models — the first run can take a few minutes on CPU.",
        expanded=True,
    )

    try:
        status.write(
            "1/4 · Loading LCPRS configuration..."
        )
        config = load_config()

        status.write(
            "2/4 · Loading PaddleOCR..."
        )
        ocr = load_ocr()

        status.write(
            "3/4 · Loading GLiNER2-PII..."
        )
        gliner = load_gliner()

        status.write(
            "4/4 · Loading PP-DocLayout-L..."
        )
        layout_model = load_layout()

        status.update(
            label="All analysis models are ready.",
            state="complete",
        )

    except Exception as error:
        status.update(
            label="Model loading failed.",
            state="error",
        )
        st.exception(error)
        st.session_state.analysis_running = False
        st.stop()

    results = []

    # IMPORTANT:
    # Temporary files are used only while the models process pages.
    # The final rendered PNG is copied into a PIL Image BEFORE the
    # temporary directory is deleted.
    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        for page_number, page in enumerate(
            pages,
            start=1,
        ):

            md(
                f"### Page {page_number} / {len(pages)}"
            )

            page_path = (
                temp_dir /
                f"page_{page_number}.png"
            )

            final_path = (
                temp_dir /
                f"page_{page_number}_final.png"
            )

            page.save(
                page_path,
                format="PNG",
            )

            progress = st.progress(
                0,
                text="OCR: extracting text and word boxes...",
            )

            try:

                # ==================================================
                # OCR
                # ==================================================

                ocr_results = list(
                    ocr.predict(
                        input=str(page_path)
                    )
                )

                if not ocr_results:
                    raise RuntimeError(
                        "PaddleOCR returned no result."
                    )

                text, words = ocr_words(
                    ocr_results[0]
                )

                coordinate_words = [
                    word
                    for word in words
                    if word.get("bbox") is not None
                ]

                if not coordinate_words:
                    raise RuntimeError(
                        "No OCR word-level bounding boxes were produced."
                    )

                progress.progress(
                    20,
                    text=(
                        f"OCR complete · "
                        f"{len(coordinate_words)} word boxes"
                    ),
                )

                # ==================================================
                # GLINER2
                # ==================================================

                gliner_result = (
                    gliner.extract_entities(
                        text,
                        config["labels"],
                        threshold=GLINER_THRESHOLD,
                        include_confidence=True,
                        include_spans=True,
                    )
                )

                detected = flatten_gliner(
                    gliner_result
                )

                predictions = []

                for entity in detected:

                    matched_words = find_words(
                        entity["text"],
                        coordinate_words,
                    )

                    box = union_box(
                        matched_words
                    )

                    if box is None:
                        continue

                    predictions.append(
                        {
                            "text": entity["text"],
                            "type": entity["type"],
                            "score": entity["score"],
                            "predicted_bbox": box,
                            "ocr_words": [
                                word["text"]
                                for word in matched_words
                            ],
                        }
                    )

                # Conservative ID-field recovery for cases where GLiNER misses an
                # explicit FIRST NAME field on structured identity documents.
                predictions = recover_name_from_id_fields(
                    coordinate_words,
                    predictions,
                )

                progress.progress(
                    45,
                    text=(
                        f"GLiNER2-PII complete · "
                        f"{len(predictions)} mapped detections"
                    ),
                )

                # ==================================================
                # PP-DOCLAYOUT
                # ==================================================

                layout_results = list(
                    layout_model.predict(
                        input=str(page_path),
                        batch_size=1,
                    )
                )

                if not layout_results:
                    raise RuntimeError(
                        "PP-DocLayout returned no result."
                    )

                regions = extract_layout(
                    layout_results[0]
                )

                document_elements = []

                qr_boxes = find_qr_codes(page)
                for qr_box in qr_boxes:
                    document_elements.append({
                        "label": "QR CODE",
                        "bbox": qr_box,
                    })

                pan_box = find_pan_box(coordinate_words)
                if pan_box is not None:
                    document_elements.append({
                        "label": "GOVERNMENT ID",
                        "bbox": pan_box,
                    })

                signature_box = find_signature_region(
                    coordinate_words,
                    regions,
                )
                if signature_box is not None:
                    document_elements.append({
                        "label": "SIGNATURE",
                        "bbox": signature_box,
                    })

                progress.progress(
                    65,
                    text=(
                        f"PP-DocLayout complete · "
                        f"{len(regions)} layout regions"
                    ),
                )

                # ==================================================
                # PII -> LAYOUT
                # ==================================================

                mappings = []

                for prediction in predictions:

                    region, method = map_to_layout(
                        prediction["predicted_bbox"],
                        regions,
                    )

                    mappings.append(
                        {
                            "pii_text": prediction["text"],
                            "pii_type": prediction["type"],
                            "confidence": prediction["score"],
                            "pii_bbox": prediction["predicted_bbox"],
                            "ocr_words": prediction["ocr_words"],

                            "layout_region_id": (
                                region["region_id"]
                                if region
                                else None
                            ),

                            "layout_region_type": (
                                region["label"]
                                if region
                                else None
                            ),

                            "layout_region_score": (
                                region["score"]
                                if region
                                else None
                            ),

                            "layout_region_bbox": (
                                region["bbox"]
                                if region
                                else None
                            ),

                            "mapping_method": method,
                        }
                    )

                mapped_count = sum(
                    item.get("layout_region_id") is not None
                    for item in mappings
                )

                progress.progress(
                    78,
                    text=(
                        f"Layout mapping complete · "
                        f"{mapped_count}/{len(mappings)} mapped"
                    ),
                )

                # ==================================================
                # LCPRS
                # ==================================================

                width, height = page.size

                risk = run_lcprs(
                    mappings,
                    width,
                    height,
                    config,
                )

                progress.progress(
                    90,
                    text=(
                        "LCPRS complete · "
                        "calculating privacy risk"
                    ),
                )

                # ==================================================
                # FINAL VISUALIZATION
                # ==================================================

                final_image(
                    page_path,
                    mappings,
                    risk,
                    final_path,
                )

                # CRITICAL FIX:
                # Copy the generated PNG into memory while the
                # temporary directory still exists.
                rendered_image = (
                    Image.open(
                        final_path
                    )
                    .convert("RGB")
                    .copy()
                )

                rendered_image = annotate_document_elements(
                    rendered_image,
                    document_elements,
                )

                progress.progress(
                    100,
                    text="Analysis complete.",
                )

                results.append(
                    {
                        "page": page_number,
                        "image": rendered_image,
                        "risk": risk,
                        "mappings": mappings,
                        "regions": regions,
                        "document_elements": document_elements,
                        "qr_count": len(qr_boxes),
                    }
                )

            except Exception as error:

                progress.empty()

                st.error(
                    f"Page {page_number} analysis failed."
                )

                st.exception(error)

    # ========================================================
    # RESULTS
    # ========================================================

    if not results:
        st.session_state.analysis_running = False
        st.error(
            "No pages could be analyzed."
        )
        st.stop()

    st.divider()

    md(
        "## Privacy Risk Assessment"
    )

    # ========================================================
    # DOCUMENT SUMMARY
    # ========================================================

    page_scores = [
        result["risk"]["normalized_lcprs_score"]
        for result in results
    ]

    average_score = (
        sum(page_scores) /
        len(page_scores)
    )

    highest_score = max(
        page_scores
    )

    highest_result = max(
        results,
        key=lambda result:
        result["risk"]["normalized_lcprs_score"],
    )

    total_pii = sum(
        result["risk"]["number_of_usable_pii"]
        for result in results
    )

    md(
        f"""
<div class="info-box">
    <b>Document:</b> {uploaded_file.name}<br>
    <b>Pages analyzed:</b> {len(results)}<br>
    <b>Total usable PII detections:</b> {total_pii}<br>
    <b>Average page LCPRS:</b> {average_score:.2f}/100<br>
    <b>Highest page score:</b> {highest_score:.2f}/100
    &nbsp; (Page {highest_result["page"]})
</div>
""",
        unsafe_allow_html=True,
    )

    # ========================================================
    # PAGE RESULTS
    # ========================================================

    for result in results:

        page_number = result["page"]
        risk = result["risk"]
        mappings = result["mappings"]
        regions = result["regions"]
        rendered_image = result["image"]

        md(
            f"### Page {page_number}"
        )

        mapped_count = sum(
            item.get("layout_region_id") is not None
            for item in mappings
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            md(
                f"""
<div class="metric-card">
    <div class="metric-label">
        LCPRS Score
    </div>
    <div class="metric-value">
        {risk["normalized_lcprs_score"]:.2f}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        with c2:
            md(
                f"""
<div class="metric-card">
    <div class="metric-label">
        Risk Category
    </div>
    <div class="metric-value">
        {risk["risk_category"]}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        with c3:
            md(
                f"""
<div class="metric-card">
    <div class="metric-label">
        PII Detected
    </div>
    <div class="metric-value">
        {risk["number_of_usable_pii"]}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        with c4:
            md(
                f"""
<div class="metric-card">
    <div class="metric-label">
        Layout Mapped
    </div>
    <div class="metric-value">
        {mapped_count}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # GOVERNANCE
        # ----------------------------------------------------

        md(
            f"""
<div class="risk-box">
    <div class="risk-box-title">
        Governance Recommendation
    </div>
    <div class="risk-box-text">
        {risk["governance_action"]}
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # FINAL DOCUMENT IMAGE
        # ----------------------------------------------------

        md(
            '<div class="section-title">Analyzed Document</div>',
            unsafe_allow_html=True,
        )

        # IMPORTANT:
        # Pass the PIL image object, NOT the deleted temporary path.
        st.image(
            rendered_image,
            width="stretch",
        )

        # ----------------------------------------------------
        # TECHNICAL DETAILS
        # ----------------------------------------------------

        with st.expander(
            "View analysis details"
        ):

            st.write(
                f"**Page:** {page_number}"
            )

            st.write(
                f"**PII detections used by LCPRS:** "
                f'{risk["number_of_usable_pii"]}'
            )

            st.write(
                f"**Layout regions:** "
                f"{len(regions)}"
            )

            st.write(
                f"**LCPRS score:** "
                f'{risk["normalized_lcprs_score"]:.2f}/100'
            )

            st.write(
                f"**Risk category:** "
                f'{risk["risk_category"]}'
            )

            if result.get("document_elements"):
                st.write("**Document elements detected:**")
                st.write(", ".join(x["label"] for x in result["document_elements"]))

            st.write(
                "**Detected PII:**"
            )

            rows = []

            for item in mappings:

                rows.append(
                    {
                        "PII Type": item.get(
                            "pii_type",
                            "",
                        ),

                        "Detected Text": item.get(
                            "pii_text",
                            "",
                        ),

                        "Confidence": round(
                            float(
                                item.get(
                                    "confidence",
                                    0,
                                )
                            ),
                            3,
                        ),

                        "Layout Region": (
                            item.get(
                                "layout_region_type"
                            )
                            or "Unmapped"
                        ),
                    }
                )

            if rows:
                st.dataframe(
                    rows,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.write(
                    "No configured PII detections were mapped."
                )

    st.session_state.analysis_running = False


# ============================================================
# FOOTER
# ============================================================

md(
    """
<div class="footer-note">
    Layout-Contextual Privacy Risk Assessment
    · Temporary PRC-2 Demonstration
    · Does not modify the Phase 1 dataset or Phase 2 experiment results
</div>
""",
    unsafe_allow_html=True,
)


