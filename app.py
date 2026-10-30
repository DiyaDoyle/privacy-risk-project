import io
import json
import re
import html as _html
import tempfile
import textwrap
import zipfile
from collections import Counter
from difflib import SequenceMatcher
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
    final_image,
    GLINER_MODEL,
    GLINER_THRESHOLD,
    LAYOUT_MODEL,
)
import lcprs_v2 as lcprs

# ============================================================
# ROBUST OCR MATCHING + EXTRA MASK TARGETS (dates, MRZ, face)
# ============================================================

# 14 JAN /JAN 13  ->  "14janjan13" once non-alphanumerics are removed
DATE_PATTERN = r"\d{1,2}[a-z]{3}(?:[a-z]{3})?\d{2,4}"

_CONFUSE = str.maketrans("oilsbzg", "0115829")


def _norm(value):
    return re.sub(r"[^a-zA-Z0-9]", "", str(value)).lower()


def _ok(word):
    box = word.get("bbox")
    return box is not None and len(box) == 4


def _rows(words):
    """Group OCR words into visual rows (by OCR line id AND by geometry)."""
    ws = [w for w in words if _ok(w)]
    if not ws:
        return []

    groups = []
    by_line = {}
    for w in ws:
        by_line.setdefault(w.get("line", 0), []).append(w)
    groups.extend(by_line.values())

    heights = sorted(float(w["bbox"][3]) - float(w["bbox"][1]) for w in ws)
    th = max(1.0, heights[len(heights) // 2])
    rows = []
    for w in sorted(ws, key=lambda w: (float(w["bbox"][1]) + float(w["bbox"][3])) / 2):
        cy = (float(w["bbox"][1]) + float(w["bbox"][3])) / 2
        for r in rows:
            if abs(cy - r["cy"]) <= th * 0.6:
                r["w"].append(w)
                r["cy"] = sum((float(x["bbox"][1]) + float(x["bbox"][3])) / 2 for x in r["w"]) / len(r["w"])
                break
        else:
            rows.append({"cy": cy, "w": [w]})
    groups.extend(r["w"] for r in rows)

    return [sorted(g, key=lambda w: float(w["bbox"][0])) for g in groups]


def _stream(row):
    chars, meta = [], []
    for wi, w in enumerate(row):
        text = str(w.get("text", ""))
        n = max(len(text), 1)
        for j, ch in enumerate(text):
            if ch.isalnum():
                chars.append(ch.lower())
                meta.append((wi, j, n))
    return "".join(chars), meta


def _span_box(row, meta, a, b):
    """Box for stream characters [a, b); slices partially covered end words."""
    first, last = meta[a], meta[b - 1]
    involved = row[first[0]: last[0] + 1]

    fx1, _, fx2, _ = map(float, row[first[0]]["bbox"])
    lx1, _, lx2, _ = map(float, row[last[0]]["bbox"])
    left = fx1 + (fx2 - fx1) * first[1] / first[2]
    right = lx1 + (lx2 - lx1) * (last[1] + 1) / last[2]

    top = min(float(w["bbox"][1]) for w in involved)
    bottom = max(float(w["bbox"][3]) for w in involved)
    if involved:
        left = min(left, fx2)
        right = max(right, lx1)
    return [left, top, right, bottom], involved


def locate_entity(entity_text, words):
    """Return (box, matched_words) or (None, [])."""
    target = _norm(entity_text)
    if not target:
        return None, []

    rows = _rows(words)
    streams = [(row, *_stream(row)) for row in rows]

    # 1) exact substring (handles partial-word overlaps)
    for row, s, meta in streams:
        idx = s.find(target)
        if idx >= 0:
            return _span_box(row, meta, idx, idx + len(target))

    # 2) OCR look-alike confusions (O/0, I/l/1, S/5 ...)
    t2 = target.translate(_CONFUSE)
    for row, s, meta in streams:
        idx = s.translate(_CONFUSE).find(t2)
        if idx >= 0:
            return _span_box(row, meta, idx, idx + len(target))

    # 3) fuzzy window (only for entities long enough to be safe)
    if len(target) >= 4:
        best, best_ratio = None, 0.85
        for row, s, meta in streams:
            for L in (len(target) - 1, len(target), len(target) + 1):
                if L < 3 or L > len(s):
                    continue
                for a in range(0, len(s) - L + 1):
                    r = SequenceMatcher(None, s[a:a + L], target).ratio()
                    if r > best_ratio:
                        best, best_ratio = (row, meta, a, a + L), r
        if best:
            row, meta, a, b = best
            return _span_box(row, meta, a, b)

    return None, []


def find_all_boxes(entity_text, words):
    """Boxes for EVERY occurrence of an entity's text on the page (exact normalised match).

    The Phase 2 matcher only returns the first occurrence, so a value repeated in
    every table row (e.g. a student name) would otherwise stay readable.
    """
    target = _norm(entity_text)
    if len(target) < 4:
        return []
    boxes, seen = [], set()
    for row in _rows(words):
        stream, meta = _stream(row)
        start = 0
        while True:
            idx = stream.find(target, start)
            if idx < 0:
                break
            box, _ = _span_box(row, meta, idx, idx + len(target))
            key = tuple(int(v / 4) for v in box)
            if key not in seen:
                seen.add(key)
                boxes.append(box)
            start = idx + len(target)
    return boxes


def boxes_overlap(a, b, ratio=0.5):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return False
    smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return smaller > 0 and (iw * ih) / smaller >= ratio


def find_regex_boxes(words, pattern=DATE_PATTERN):
    """Boxes for every regex match on the OCR character stream (deduplicated)."""
    boxes, seen = [], set()
    for row in _rows(words):
        s, meta = _stream(row)
        for m in re.finditer(pattern, s):
            box, _ = _span_box(row, meta, m.start(), m.end())
            key = tuple(int(v / 4) for v in box)
            if key not in seen:
                seen.add(key)
                boxes.append(box)
    return boxes


def find_mrz_boxes(words):
    """Machine-readable-zone lines: rows dense in '<' filler characters."""
    boxes = []
    for row in _rows(words):
        joined = "".join(str(w.get("text", "")) for w in row)
        if joined.count("<") >= 4:
            boxes.append([
                min(float(w["bbox"][0]) for w in row),
                min(float(w["bbox"][1]) for w in row),
                max(float(w["bbox"][2]) for w in row),
                max(float(w["bbox"][3]) for w in row),
            ])
    # drop duplicates produced by line-id + geometry grouping
    unique = []
    for b in boxes:
        if not any(abs(b[1] - u[1]) < 6 and abs(b[0] - u[0]) < 6 for u in unique):
            unique.append(b)
    return unique


def find_face_boxes(pil_image):
    """Frontal faces (ID photos) via OpenCV Haar cascade; boxes padded by 25%."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        return []
    gray = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(max(40, gray.shape[1] // 25),) * 2)
    h, w = gray.shape
    out = []
    for x, y, fw, fh in faces:
        px, py = fw * 0.25, fh * 0.25
        out.append([max(0, x - px), max(0, y - py), min(w, x + fw + px), min(h, y + fh + py)])
    return out


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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

#MainMenu, footer, header { visibility: hidden; }

html, body, .stApp { font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif; }
.stApp *:not([data-testid="stIconMaterial"]):not([class*="material" i]):not([class*="icon" i]) { font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif; }
/* keep Streamlit's icon font so arrows and check marks render as icons, not text */
[data-testid="stIconMaterial"], [class*="material-symbols" i], [class*="material-icons" i] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
}
.stApp {
    background:
        radial-gradient(1100px 520px at 8% -8%, #15264a 0%, rgba(11,18,32,0) 62%),
        radial-gradient(900px 480px at 100% 0%, #1a1f3d 0%, rgba(11,18,32,0) 55%),
        #0b1220;
    color: #e5e7eb;
}
.block-container { max-width: 1240px; padding-top: 1.6rem; padding-bottom: 3rem; }
h1, h2, h3, h4 { color: #f8fafc; letter-spacing: -0.01em; }

/* ---------- hero ---------- */
.hero {
    border: 1px solid #1f2b44;
    border-radius: 18px;
    padding: 30px 34px 26px 34px;
    margin-bottom: 22px;
    background: linear-gradient(135deg, rgba(37,99,235,0.18) 0%, rgba(17,27,46,0.9) 45%, rgba(17,27,46,0.9) 100%);
}
.eyebrow { font-size: 11.5px; font-weight: 700; letter-spacing: 2.2px; color: #7aa2f7; text-transform: uppercase; margin-bottom: 10px; }
.hero-title { font-size: 33px; font-weight: 800; color: #f8fafc; line-height: 1.15; letter-spacing: -0.02em; }
.hero-sub { font-size: 15px; color: #9fb0c8; max-width: 760px; margin-top: 10px; line-height: 1.55; }
.steps { display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap; }
.step { display: inline-flex; align-items: center; gap: 9px; font-size: 12.5px; font-weight: 600; color: #cbd5e1;
        background: rgba(15,23,42,0.7); border: 1px solid #243350; border-radius: 999px; padding: 6px 14px 6px 7px; }
.step span { display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%;
             background: #2563eb; color: #fff; font-size: 11px; font-weight: 700; }

/* ---------- section headings ---------- */
.section-head { display: flex; align-items: baseline; gap: 12px; margin: 30px 0 14px 0; }
.section-head .t { font-size: 19px; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em; }
.section-head .s { font-size: 13px; color: #7f91ab; }

/* ---------- cards ---------- */
.file-card { display: flex; align-items: center; gap: 14px; border: 1px solid #1f2b44; border-radius: 14px;
             padding: 13px 18px; background: #111b2e; margin: 14px 0 16px 0; }
.file-icon { width: 38px; height: 38px; border-radius: 10px; background: rgba(37,99,235,0.18); color: #7aa2f7;
             display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 800; letter-spacing: 0.5px; }
.file-name { font-size: 14.5px; font-weight: 600; color: #f1f5f9; }
.file-meta { font-size: 12px; color: #7f91ab; margin-top: 2px; }

.kpi-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
.kpi-grid.four { grid-template-columns: repeat(4, 1fr); }
@media (max-width: 900px) { .kpi-grid.four { grid-template-columns: repeat(2, 1fr); } }
.kpi { border: 1px solid #1f2b44; border-left: 3px solid var(--accent, #3b82f6); border-radius: 12px;
       padding: 15px 18px; background: #111b2e; min-height: 92px; }
.kpi-label { font-size: 10.5px; font-weight: 700; letter-spacing: 1.1px; color: #7f91ab; text-transform: uppercase; }
.kpi-value { font-size: 26px; font-weight: 750; color: #f8fafc; margin-top: 6px; line-height: 1.1; }
.kpi-sub { font-size: 11.5px; color: #7f91ab; margin-top: 4px; min-height: 14px; }

.score-panel { border: 1px solid #1f2b44; border-radius: 16px; padding: 22px 26px; background: #111b2e; height: 100%; }
.score-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; }
.score-num { font-size: 52px; font-weight: 800; line-height: 1; color: #f8fafc; letter-spacing: -0.03em; }
.score-num small { font-size: 18px; font-weight: 600; color: #7f91ab; margin-left: 4px; }
.badge { display: inline-block; font-size: 11.5px; font-weight: 800; letter-spacing: 1.2px; text-transform: uppercase;
         padding: 6px 13px; border-radius: 999px; color: #fff; }
.gauge-track { height: 10px; border-radius: 999px; margin-top: 20px; position: relative; overflow: hidden;
               background: linear-gradient(90deg, #14532d 0%, #854d0e 40%, #9a3412 70%, #7f1d1d 100%); opacity: 0.55; }
.gauge-wrap { position: relative; }
.gauge-marker { position: absolute; top: 12px; width: 4px; height: 18px; border-radius: 3px; background: #fff;
                box-shadow: 0 0 0 3px rgba(255,255,255,0.18); transform: translateX(-2px); }
.gauge-scale { display: flex; justify-content: space-between; font-size: 10.5px; color: #64748b; margin-top: 16px; }
.gov-title { font-size: 10.5px; font-weight: 700; letter-spacing: 1.1px; color: #7f91ab; text-transform: uppercase; margin-top: 18px; }
.gov-text { font-size: 14px; color: #dbe4f0; line-height: 1.55; margin-top: 6px; }

.notice { border-radius: 12px; padding: 13px 17px; font-size: 13.5px; line-height: 1.55; margin: 12px 0; border: 1px solid; }
.notice.warn { background: rgba(245,158,11,0.09); border-color: rgba(245,158,11,0.35); color: #fcd9a0; }
.notice.info { background: rgba(59,130,246,0.09); border-color: rgba(59,130,246,0.35); color: #bcd3fb; }
.notice.ok   { background: rgba(34,197,94,0.09);  border-color: rgba(34,197,94,0.35);  color: #b4efc9; }

.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 14px 0; }
.chip { font-size: 12px; font-weight: 600; color: #cfe0ff; background: rgba(37,99,235,0.14); border: 1px solid rgba(96,150,255,0.28);
        border-radius: 999px; padding: 4px 12px; }
.chip b { color: #fff; margin-left: 5px; }

.dl-title { font-size: 15px; font-weight: 700; color: #f8fafc; }
.dl-desc { font-size: 12.5px; color: #8ea0b9; line-height: 1.5; margin: 4px 0 12px 0; min-height: 38px; }

.footer-note { text-align: center; color: #5d6f8a; font-size: 11.5px; margin-top: 46px; padding-top: 18px; border-top: 1px solid #1a2438; line-height: 1.6; }

/* ---------- streamlit widgets ---------- */
[data-testid="stFileUploaderDropzone"] { background: #111b2e; border: 1.5px dashed #34507d; border-radius: 14px; padding: 26px; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #5b8def; background: #13203a; }
[data-testid="stFileUploader"] label p { color: #cbd5e1; font-weight: 600; }

div.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
    min-height: 48px; border-radius: 11px; font-weight: 700; font-size: 15px; border: none; color: #fff;
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%); box-shadow: 0 6px 18px rgba(37,99,235,0.28);
}
div.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { filter: brightness(1.1); }
[data-testid="stDownloadButton"] button { border-radius: 10px; font-weight: 600; min-height: 44px; }
[data-testid="stDownloadButton"] button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
    background: #16233b; border: 1px solid #2a3b5c; color: #e2e8f0;
}
[data-testid="stDownloadButton"] button[kind="secondary"]:hover { border-color: #5b8def; color: #fff; }

div[data-testid="stVerticalBlockBorderWrapper"] { background: #111b2e; border-color: #1f2b44 !important; border-radius: 14px; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #1f2b44; }
.stTabs [data-baseweb="tab"] { height: 44px; padding: 0 16px; color: #8ea0b9; font-weight: 600; background: transparent; }
.stTabs [aria-selected="true"] { color: #f8fafc; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #3b82f6; height: 3px; }

[data-testid="stDataFrame"] { border: 1px solid #1f2b44; border-radius: 12px; overflow: hidden; }
[data-testid="stImage"] img { border-radius: 12px; border: 1px solid #1f2b44; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

md(
    """
<div class="hero">
<div class="eyebrow">Privacy Analytics</div>
<div class="hero-title">Layout-Contextual Privacy Risk Assessment</div>
<div class="hero-sub">Detect sensitive information in documents, understand where it sits on the page, score the privacy risk, and export a masked copy with a full audit trail.</div>
<div class="steps"><div class="step"><span>1</span>Upload</div><div class="step"><span>2</span>Analyze</div><div class="step"><span>3</span>Review &amp; download</div></div>
</div>
""",
    unsafe_allow_html=True,
)


def esc(value):
    return _html.escape(str(value))


def section(title, subtitle=""):
    md(
        f'<div class="section-head"><span class="t">{esc(title)}</span>'
        f'<span class="s">{esc(subtitle)}</span></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# UPLOAD
# ============================================================

section("Document", "PDF or image · every page is analyzed")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=[
        "pdf",
        "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif",
        "ppm", "pgm", "pbm", "pnm", "ico",
        "jp2", "j2k", "jpf", "jpx", "jpm", "mj2",
        "avif",
    ],
    label_visibility="collapsed",
    help="Supported: PDF, PNG, JPG/JPEG, WEBP, BMP, TIFF, GIF, PPM/PGM/PBM/PNM, ICO, JPEG 2000 and AVIF.",
)
if uploaded_file is None:
    md(
        '<div class="notice info">Select a document to begin. Supported formats: PDF, PNG, JPG, '
        "WEBP, BMP, TIFF, GIF, JPEG 2000 and AVIF.</div>",
        unsafe_allow_html=True,
    )
    st.stop()


# Read the uploaded file ONCE.
file_bytes = uploaded_file.getvalue()
file_size_mb = len(file_bytes) / (1024 * 1024)
file_ext = Path(uploaded_file.name).suffix.lstrip(".").upper()[:4] or "FILE"

md(
    f"""
<div class="file-card">
<div class="file-icon">{esc(file_ext)}</div>
<div><div class="file-name">{esc(uploaded_file.name)}</div>
<div class="file-meta">{file_size_mb:.2f} MB</div></div>
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
    config = load_configuration()
    # Validate the v3 formula/config used by process_new_document.py.
    lcprs.validate_configuration()
    return config


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


def find_words_for_app(entity_text, words):
    """Match entity text to OCR boxes conservatively, including nearby split-line tokens.

    The Phase 2 matcher remains primary. Fallbacks only accept exact normalized text;
    fuzzy matches are deliberately avoided for PII so unrelated text is not masked.
    """
    matched = find_words(entity_text, words)
    if matched:
        return matched

    target = simple_norm(entity_text)
    if not target:
        return []

    valid_words = [
        w for w in words
        if w.get("bbox") is not None and len(w.get("bbox", [])) == 4
    ]
    if not valid_words:
        return []

    # First: exact concatenation of adjacent words on the same OCR line.
    by_line = {}
    for word in valid_words:
        by_line.setdefault(word.get("line", 0), []).append(word)

    candidate_groups = []
    for line_words in by_line.values():
        candidate_groups.append(sorted(line_words, key=lambda w: float(w["bbox"][0])))

    # Second: OCR sometimes assigns a date field to separate line IDs despite the
    # boxes being visually adjacent. Build small geometric rows using box centers.
    heights = [max(1.0, float(w["bbox"][3]) - float(w["bbox"][1])) for w in valid_words]
    typical_height = sorted(heights)[len(heights) // 2]
    geometric_rows = []
    for word in sorted(valid_words, key=lambda w: ((float(w["bbox"][1]) + float(w["bbox"][3])) / 2, float(w["bbox"][0]))):
        x1, y1, x2, y2 = map(float, word["bbox"])
        cy = (y1 + y2) / 2
        best_row = None
        best_delta = float("inf")
        for row in geometric_rows:
            row_cy = sum((float(w["bbox"][1]) + float(w["bbox"][3])) / 2 for w in row) / len(row)
            delta = abs(cy - row_cy)
            if delta <= max(3.0, typical_height * 0.55) and delta < best_delta:
                best_row, best_delta = row, delta
        if best_row is None:
            geometric_rows.append([word])
        else:
            best_row.append(word)
    candidate_groups.extend(sorted(row, key=lambda w: float(w["bbox"][0])) for row in geometric_rows)

    seen = set()
    for group in candidate_groups:
        for start in range(len(group)):
            combined = ""
            selected = []
            previous = None
            for end in range(start, min(len(group), start + 12)):
                word = group[end]
                x1, y1, x2, y2 = map(float, word["bbox"])
                if previous is not None:
                    px1, py1, px2, py2 = map(float, previous["bbox"])
                    # Do not combine distant tokens; only consider visually adjacent words.
                    gap = x1 - px2
                    if gap > max(12.0, typical_height * 1.5):
                        break
                    if gap < -typical_height * 0.5:
                        continue
                token = simple_norm(word.get("text", ""))
                if not token:
                    previous = word
                    continue
                combined += token
                selected.append(word)
                if combined == target:
                    key = tuple(id(w) for w in selected)
                    if key not in seen:
                        return selected
                if len(combined) > len(target):
                    break
                previous = word
    return []


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
# RESULT BUILDERS (computed once, stored in session state)
# ============================================================

RISK_COLORS = {
    "low": "#16a34a",
    "minimal": "#16a34a",
    "medium": "#d97706",
    "moderate": "#d97706",
    "high": "#ea580c",
    "critical": "#dc2626",
}


def risk_color(category):
    return RISK_COLORS.get(str(category).strip().lower(), "#475569")


def pretty(value):
    text = str(value).replace("_", " ").strip().title()
    for small in (" Of ", " And ", " Or ", " To ", " The "):
        text = text.replace(small, small.lower())
    return text


def kpi(label, value, sub="", accent="#3b82f6"):
    return (
        f'<div class="kpi" style="--accent:{accent}"><div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{esc(value)}</div><div class="kpi-sub">{esc(sub)}</div></div>'
    )


def dl_button(label, data, file_name, mime, key, primary=False):
    """Download button that never triggers a full-page rerun when Streamlit supports it."""
    kwargs = dict(
        label=label, data=data, file_name=file_name, mime=mime, key=key,
        type="primary" if primary else "secondary",
    )
    for extra in (
        {"on_click": "ignore", "width": "stretch"},
        {"on_click": "ignore", "use_container_width": True},
        {"use_container_width": True},
        {},
    ):
        try:
            return st.download_button(**kwargs, **extra)
        except TypeError:
            continue



EXTRA_LABELS = {"DATE": "Dates", "MRZ": "ID machine-readable lines", "FACE": "Faces", "REPEAT": "Repeated values"}


def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def build_report_pdf(filename, results, audit_pages, block_reason, stats):
    """Plain-language PDF summary. It lists PII *types and counts* only, never the values."""
    from datetime import datetime

    W, H, M = 595, 842, 54
    ink, muted, blue = (0.10, 0.14, 0.22), (0.38, 0.44, 0.53), (0.15, 0.39, 0.92)
    doc = pymupdf.open()
    state = {"page": None, "y": 0}

    def clean(t):
        return str(t).encode("latin-1", "replace").decode("latin-1")

    def new_page():
        state["page"] = doc.new_page(width=W, height=H)
        state["y"] = M

    def ensure(height):
        if state["page"] is None or state["y"] + height > H - M:
            new_page()

    def text(t, size=10.5, bold=False, color=ink, indent=0, gap=5):
        font = "hebo" if bold else "helv"
        width = W - 2 * M - indent
        lines, cur = [], ""
        for word in clean(t).split():
            trial = (cur + " " + word).strip()
            if pymupdf.get_text_length(trial, fontname=font, fontsize=size) <= width:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        lh = size * 1.45
        for line in lines:
            ensure(lh)
            state["page"].insert_text((M + indent, state["y"] + size), line, fontname=font, fontsize=size, color=color)
            state["y"] += lh
        state["y"] += gap

    def heading(t):
        ensure(40)
        state["y"] += 8
        text(t, 13.5, True, blue, gap=2)
        state["page"].draw_line((M, state["y"]), (W - M, state["y"]), color=(0.85, 0.88, 0.93), width=0.8)
        state["y"] += 8

    # ---- title band
    new_page()
    page = state["page"]
    page.draw_rect(pymupdf.Rect(0, 0, W, 104), color=None, fill=(0.07, 0.11, 0.2))
    page.insert_text((M, 48), "Privacy Risk Assessment Report", fontname="hebo", fontsize=22, color=(1, 1, 1))
    page.insert_text((M, 70), clean(f"Document: {filename}"), fontname="helv", fontsize=10.5, color=(0.78, 0.84, 0.93))
    page.insert_text((M, 87), f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}", fontname="helv", fontsize=9.5, color=(0.6, 0.68, 0.8))
    state["y"] = 126

    # ---- overall result
    category = stats["category"] or "Not scored"
    color = _rgb(risk_color(category))
    ensure(60)
    top = state["y"]
    state["page"].draw_rect(pymupdf.Rect(M, top, W - M, top + 56), color=None, fill=color, radius=0.08)
    state["page"].insert_text((M + 16, top + 24), clean(f"Overall privacy risk: {str(category).upper()}"), fontname="hebo", fontsize=15, color=(1, 1, 1))
    score_line = "Not scored" if stats["highest"] is None else f"Highest page score {stats['highest']:.2f} out of 100 (page {stats['page']})"
    state["page"].insert_text((M + 16, top + 43), clean(score_line), fontname="helv", fontsize=10.5, color=(1, 1, 1))
    state["y"] = top + 74

    heading("Summary")
    text(f"We analysed {len(results)} page(s) and found {stats['total_pii']} piece(s) of personal or sensitive information. "
         "Each page is scored from 0 (very low risk) to 100 (very high risk) based on what was found and how close together the items sit on the page.")

    gov = ""
    scored = [r for r in results if r["risk"]["normalized_lcprs_score"] is not None]
    if scored:
        gov = max(scored, key=lambda r: r["risk"]["normalized_lcprs_score"])["risk"]["governance_action"]
    if gov:
        heading("What we recommend")
        text(gov)

    heading("What was found")
    for result, audit in zip(results, audit_pages):
        risk = result["risk"]
        score = risk["normalized_lcprs_score"]
        score_txt = "not scored" if score is None else f"{score:.2f} / 100"
        text(f"Page {result['page']}  -  {risk['risk_category']} risk ({score_txt})", 11, True, gap=2)
        counts = Counter(pretty(f.get("pii_type", "unknown")) for f in (result.get("findings") or result["mappings"]))
        if counts:
            for kind, n in counts.most_common():
                text(f"-  {kind}: {n}", indent=12, gap=0)
            state["y"] += 4
        else:
            text("No sensitive items were located on this page.", indent=12)
        extra = audit.get("extra_regions_masked", {})
        if extra:
            text("Also covered automatically: " + ", ".join(f"{EXTRA_LABELS.get(k, k)} ({v})" for k, v in extra.items()) + ".",
                 color=muted, indent=12)

    heading("What was masked")
    if block_reason:
        text("A masked copy was NOT produced for download.", bold=True, color=(0.7, 0.25, 0.05), gap=2)
        text(block_reason)
    else:
        masked_total = sum(a.get("entities_masked", 0) for a in audit_pages)
        text(f"A masked copy was produced. {masked_total} detected item(s) were covered with solid black boxes, "
             "together with any dates, ID machine-readable lines and faces found on the pages. "
             "The masked file contains page images only, so the original text cannot be copied out of it.")

    unmapped_rows = [(r["page"], u) for r in results for u in r.get("unmapped_entities", [])]
    if unmapped_rows:
        heading("Items that need a manual check")
        text("These items were detected but could not be located on the page, so they were not scored or masked automatically:")
        for pg, u in unmapped_rows:
            text(f"-  Page {pg}: {pretty(u.get('pii_type', 'unknown'))}", indent=12, gap=0)
        state["y"] += 4

    heading("Before you share anything")
    for tip in (
        "Open the masked PDF and look at every page. Automatic detection can miss information or flag harmless text.",
        "Keep the original document secure. This report and the masked copy do not replace it.",
        "The risk score is an estimate to help you decide how carefully to handle the document, not proof that it is safe.",
    ):
        text("-  " + tip, indent=12, gap=3)

    total = len(doc)
    for i, p in enumerate(doc, start=1):
        p.insert_text((M, H - 28), f"Privacy Risk Assessment Report  |  Page {i} of {total}", fontname="helv", fontsize=8.5, color=muted)

    data = doc.tobytes()
    doc.close()
    return data


def build_outputs(results, pages, config, filename):
    scored = [r for r in results if r["risk"]["normalized_lcprs_score"] is not None]
    page_scores = [r["risk"]["normalized_lcprs_score"] for r in scored]
    average_score = (sum(page_scores) / len(page_scores)) if page_scores else None
    highest_score = max(page_scores) if page_scores else None
    highest_result = max(scored, key=lambda r: r["risk"]["normalized_lcprs_score"]) if scored else None
    total_pii = sum(r["risk"]["number_of_usable_pii"] for r in results)
    total_unmapped = sum(len(r.get("unmapped_entities", [])) for r in results)

    masking_config = config.get("masking", {})
    block_reason = None
    if not masking_config.get("enabled", True):
        block_reason = "Masking is disabled in config/lcprs_config.json, so no masked PDF was generated."
    elif total_unmapped:
        block_reason = (
            "At least one detected PII entity could not be matched to an OCR bounding box. "
            "The masked PDF is withheld so it is never presented as completely masked. "
            "See the assessment report for details."
        )

    from PIL import ImageDraw

    padding = int(masking_config.get("bbox_padding_pixels", 2))
    fill_color = tuple(masking_config.get("fill_color", [0, 0, 0]))

    masked_pages, audit_pages, page_report_items = [], [], []

    for result in results:
        page_number = result["page"]
        mappings = result["mappings"]
        invalid_boxes = []
        masked = pages[page_number - 1].copy().convert("RGB")
        draw = ImageDraw.Draw(masked)
        masked_count = 0
        masked_types = {}
        drawn_boxes = set()

        for index, item in enumerate(mappings):
            bbox = item.get("pii_bbox")
            try:
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    raise ValueError("missing or malformed bbox")
                x1, y1, x2, y2 = [float(v) for v in bbox]
                if not all(np.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                    raise ValueError("invalid bbox coordinates")
                left = max(0, min(masked.width, int(x1) - padding))
                top = max(0, min(masked.height, int(y1) - padding))
                right = max(0, min(masked.width, int(x2 + padding + 0.999)))
                bottom = max(0, min(masked.height, int(y2 + padding + 0.999)))
                if right <= left or bottom <= top:
                    raise ValueError("bbox outside page bounds")
                draw.rectangle([left, top, right, bottom], fill=fill_color)
                box_key = tuple(int(v / 3) for v in (left, top, right, bottom))
                if box_key not in drawn_boxes:
                    drawn_boxes.add(box_key)
                    masked_count += 1
                    pii_type = str(item.get("pii_type", "unknown"))
                    masked_types[pii_type] = masked_types.get(pii_type, 0) + 1
            except (TypeError, ValueError, OverflowError):
                invalid_boxes.append(index)

        # Dates, MRZ lines and faces are always blacked out.
        extra_masked = {}
        for extra in result.get("extra_mask_boxes", []):
            try:
                x1, y1, x2, y2 = [float(v) for v in extra["bbox"]]
                left = max(0, int(x1) - padding)
                top = max(0, int(y1) - padding)
                right = min(masked.width, int(x2 + padding + 0.999))
                bottom = min(masked.height, int(y2 + padding + 0.999))
                if right > left and bottom > top:
                    draw.rectangle([left, top, right, bottom], fill=fill_color)
                    extra_masked[extra["label"]] = extra_masked.get(extra["label"], 0) + 1
            except (TypeError, ValueError, KeyError, OverflowError):
                pass

        unmapped = result.get("unmapped_entities", [])
        audit_pages.append({
            "page": page_number,
            "source_file": filename,
            "extra_regions_masked": extra_masked,
            "detector_entities_found": result["risk"].get("detected_entity_count", len(mappings) + len(unmapped)),
            "entities_mapped_to_boxes": len(mappings),
            "entities_without_boxes": [
                {"pii_type": u.get("pii_type", "unknown"), "confidence": u.get("confidence"), "reason": u.get("reason")}
                for u in unmapped
            ],
            "entities_masked": masked_count,
            "entities_with_invalid_bbox": len(invalid_boxes),
            "masked_types": masked_types,
            "masking_method": "Opaque rectangles on rasterized page; original PDF text objects are not copied.",
            "masking_complete_for_detected_entities": (
                not unmapped and not invalid_boxes
            ),
        })
        page_report_items.append({
            "page": page_number,
            "lcprs_score": result["risk"].get("normalized_lcprs_score"),
            "risk_category": result["risk"].get("risk_category"),
            "detected_pii_count": result["risk"].get("detected_entity_count", len(mappings)),
            "mapped_pii_count": len(mappings),
            "unmapped_pii_count": len(unmapped),
            "layout_regions_detected": len(result.get("regions", [])),
            "document_elements_detected": [x.get("label") for x in result.get("document_elements", [])],
        })
        masked_pages.append(masked)

    if any(a["entities_with_invalid_bbox"] for a in audit_pages):
        block_reason = "One or more PII bounding boxes were invalid. Masked PDF generation is blocked."

    pdf_bytes = None
    if not block_reason:
        buffer = io.BytesIO()
        masked_pages[0].save(
            buffer, format="PDF", save_all=True, append_images=masked_pages[1:], resolution=150.0
        )
        pdf_bytes = buffer.getvalue()

    report_pdf = build_report_pdf(
        filename, results, audit_pages, block_reason,
        {"average": average_score, "highest": highest_score, "page": highest_result["page"] if highest_result else None,
         "category": highest_result["risk"]["risk_category"] if highest_result else None, "total_pii": total_pii},
    )

    return {
        "average_score": average_score,
        "highest_score": highest_score,
        "highest_page": highest_result["page"] if highest_result else None,
        "highest_category": highest_result["risk"]["risk_category"] if highest_result else None,
        "total_pii": total_pii,
        "total_unmapped": total_unmapped,
        "block_reason": block_reason,
        "masked_pages": masked_pages,
        "audit_pages": audit_pages,
        "pdf_bytes": pdf_bytes,
        "report_pdf": report_pdf,
    }


# ============================================================
# RESULT RENDERING (reads only from session state)
# ============================================================

def render_page(result, masked_image, audit, block_reason, key):
    risk = result["risk"]
    mappings = result["mappings"]
    regions = result["regions"]
    unmapped = result.get("unmapped_entities", [])
    score = risk["normalized_lcprs_score"]
    category = risk["risk_category"]
    color = risk_color(category)
    mapped_count = sum(item.get("layout_region_id") is not None for item in mappings)
    extra_total = sum(audit.get("extra_regions_masked", {}).values())

    pct = 0.0 if score is None else max(0.0, min(100.0, float(score)))
    score_text = "N/A" if score is None else f"{score:.2f}"

    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        md(
            f'<div class="score-panel">'
            f'<div class="score-top"><div><div class="kpi-label">Privacy risk score</div>'
            f'<div class="score-num">{esc(score_text)}<small>/ 100</small></div></div>'
            f'<span class="badge" style="background:{color}">{esc(category)}</span></div>'
            f'<div class="gauge-wrap"><div class="gauge-track"></div>'
            f'<div class="gauge-marker" style="left:{pct:.1f}%"></div></div>'
            f'<div class="gauge-scale"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>'
            f'<div class="gov-title">Governance recommendation</div>'
            f'<div class="gov-text">{esc(risk["governance_action"])}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with right:
        md(
            '<div class="kpi-grid">'
            + kpi("PII detected", risk["number_of_usable_pii"],
                (f'{risk["repeated_values"]} repeat(s) counted once' if risk.get("repeated_values") else "distinct sensitive items"), "#3b82f6")
            + kpi("Items masked", audit.get("entities_masked", 0), "covered in masked copy", "#22c55e")
            + kpi("Extra areas masked", extra_total, "dates · ID lines · faces", "#f59e0b")
            + kpi("Needs review", len(unmapped), "could not be located", "#ef4444" if unmapped else "#64748b")
            + "</div>",
            unsafe_allow_html=True,
        )

    if unmapped:
        md(
            f'<div class="notice warn"><b>{len(unmapped)} detected item(s)</b> could not be matched to OCR boxes '
            "and were not scored. Review this page manually; the score may underestimate risk.</div>",
            unsafe_allow_html=True,
        )

    tab_doc, tab_mask, tab_find = st.tabs(["Analyzed document", "Masked preview", "Findings"])

    with tab_doc:
        st.image(result["image"], width="stretch")

    with tab_mask:
        if block_reason:
            md(
                '<div class="notice warn">Preview only. Masking is incomplete for this document, '
                "so the masked PDF is withheld.</div>",
                unsafe_allow_html=True,
            )
        else:
            md(
                '<div class="notice ok">Detected PII, dates, ID machine-readable lines and faces are blacked out. '
                "Check this preview before sharing.</div>",
                unsafe_allow_html=True,
            )
        st.image(masked_image, width="stretch")

    with tab_find:
        findings = result.get("findings") or [
            {"pii_type": m.get("pii_type", ""), "pii_text": m.get("pii_text", ""), "count": 1} for m in mappings
        ]
        counts = Counter(str(f.get("pii_type", "unknown")) for f in findings)
        if counts:
            md(
                '<div class="chips">'
                + "".join(f'<span class="chip">{esc(pretty(t))}<b>{n}</b></span>' for t, n in counts.most_common())
                + "</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                [
                    {"PII Type": pretty(f.get("pii_type", "")), "Detected Text": f.get("pii_text", ""),
                     "Times found": int(f.get("count", 1))}
                    for f in findings
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.write("No configured PII detections were mapped.")

        extra = audit.get("extra_regions_masked", {})
        if extra:
            md('<div class="section-head"><span class="t" style="font-size:15px">Also masked automatically</span></div>', unsafe_allow_html=True)
            md(
                '<div class="chips">'
                + "".join(f'<span class="chip">{esc(EXTRA_LABELS.get(k, k))}<b>{v}</b></span>' for k, v in extra.items())
                + "</div>",
                unsafe_allow_html=True,
            )

        if unmapped:
            md('<div class="section-head"><span class="t" style="font-size:15px">Needs a manual check</span></div>', unsafe_allow_html=True)
            st.dataframe(
                [{"PII Type": pretty(u.get("pii_type")), "Issue": "Could not be located on the page"} for u in unmapped],
                width="stretch",
                hide_index=True,
            )


def render_results(a):
    results = a["results"]
    block_reason = a["block_reason"]

    section("Risk summary", a["file_name"])
    avg = "N/A" if a["average_score"] is None else f'{a["average_score"]:.2f}'
    high = "N/A" if a["highest_score"] is None else f'{a["highest_score"]:.2f}'
    top_cat = a.get("highest_category")
    md(
        '<div class="kpi-grid four">'
        + kpi("Pages analyzed", len(results), "", "#3b82f6")
        + kpi("PII detections", a["total_pii"], f'{a["total_unmapped"]} without a box', "#8b5cf6")
        + kpi("Average risk score", avg, "out of 100", "#06b6d4")
        + kpi(
            "Highest page score",
            high,
            (f'Page {a["highest_page"]} · {top_cat}' if a["highest_page"] else ""),
            risk_color(top_cat) if top_cat else "#475569",
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    section("Downloads", "your results stay here after downloading")
    c1, c2 = st.columns(2, gap="medium")
    stem = Path(a["file_name"]).stem
    with c1:
        with st.container(border=True):
            md(
                '<div class="dl-title">Masked PDF</div>'
                '<div class="dl-desc">A copy of your document with sensitive information blacked out.</div>',
                unsafe_allow_html=True,
            )
            if a["pdf_bytes"]:
                dl_button("Download masked PDF", a["pdf_bytes"], f"{stem}_masked.pdf",
                          "application/pdf", "dl_pdf", primary=True)
            else:
                md('<div class="notice warn" style="margin:0">Withheld. See the note below.</div>', unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            md(
                '<div class="dl-title">Assessment report</div>'
                '<div class="dl-desc">A plain-language PDF summary: risk level, what was found and what was masked. '
                "It never repeats the sensitive values.</div>",
                unsafe_allow_html=True,
            )
            dl_button("Download assessment report (PDF)", a["report_pdf"], f"{stem}_assessment_report.pdf",
                      "application/pdf", "dl_report")

    if block_reason:
        md(f'<div class="notice warn">{esc(block_reason)}</div>', unsafe_allow_html=True)
    else:
        md(
            '<div class="notice ok">Masked PDF and assessment report are ready. '
            "Review the masked pages before sharing the document.</div>",
            unsafe_allow_html=True,
        )

    section("Page analysis", f"{len(results)} page(s)")
    if len(results) > 1:
        tabs = st.tabs([f"Page {r['page']}" for r in results])
        for index, (tab, result) in enumerate(zip(tabs, results)):
            with tab:
                render_page(result, a["masked_pages"][index], a["audit_pages"][index], block_reason, f"p{index}")
    else:
        render_page(results[0], a["masked_pages"][0], a["audit_pages"][0], block_reason, "p0")



# ============================================================
# ANALYZE
# ============================================================

file_key = f"{uploaded_file.name}:{len(file_bytes)}"
if st.session_state.get("file_key") != file_key:
    st.session_state.file_key = file_key
    st.session_state.pop("analysis", None)

analyze_clicked = st.button(
    "Analyze document",
    type="primary",
    width="stretch",
)

if analyze_clicked:
    st.session_state.pop("analysis", None)

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
                unmapped_entities = []

                # Date-shaped OCR spans (e.g. "14 JAN /JAN 13"). Used as a fallback
                # for date entities and always masked in the output.
                date_boxes = find_regex_boxes(coordinate_words)
                used_date_boxes = set()

                for entity in detected:

                    matched_words = find_words_for_app(
                        entity["text"],
                        coordinate_words,
                    )

                    box = union_box(
                        matched_words
                    )

                    # Fallback 1: character-stream matching (partial words, OCR look-alikes).
                    if box is None:
                        box, matched_words = locate_entity(
                            entity["text"],
                            coordinate_words,
                        )

                    # Fallback 2: date entities -> next unused date-shaped OCR span.
                    if box is None and "date" in str(entity.get("type", "")).lower():
                        for d_index, d_box in enumerate(date_boxes):
                            if d_index not in used_date_boxes:
                                used_date_boxes.add(d_index)
                                box, matched_words = d_box, []
                                break

                    if box is None:
                        # Keep an audit record and show a coverage warning below.
                        # These detections cannot be included in spatial scoring or masking.
                        unmapped_entities.append({
                            "pii_type": entity.get("type", "unknown"),
                            "confidence": entity.get("score"),
                            "reason": "Could not match detected text to OCR word boxes",
                        })
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

                # Extra regions that are always blacked out in the masked PDF.
                extra_mask_boxes = [{"label": "DATE", "bbox": b} for b in date_boxes]
                for mrz_box in find_mrz_boxes(coordinate_words):
                    extra_mask_boxes.append({"label": "MRZ", "bbox": mrz_box})
                    document_elements.append({"label": "MRZ", "bbox": mrz_box})
                for face_box in find_face_boxes(page):
                    extra_mask_boxes.append({"label": "FACE", "bbox": face_box})
                    document_elements.append({"label": "FACE", "bbox": face_box})

                # A value printed more than once (a name in every table row, say) is
                # boxed only at its first occurrence by the matcher above. Find the
                # other occurrences so they are masked too.
                unique_entities = {}
                for prediction in predictions:
                    entity_key = (str(prediction["type"]).lower(), simple_norm(prediction["text"]))
                    unique_entities.setdefault(entity_key, prediction)
                repeat_counts = {}
                for entity_key, prediction in unique_entities.items():
                    for occurrence in find_all_boxes(prediction["text"], coordinate_words):
                        if any(boxes_overlap(occurrence, q["predicted_bbox"]) for q in predictions):
                            continue
                        extra_mask_boxes.append({"label": "REPEAT", "bbox": occurrence})
                        repeat_counts[entity_key] = repeat_counts.get(entity_key, 0) + 1

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

                # Score each distinct value once. The same name or ID printed in every
                # table row is one exposure, not many, and counting repeats makes the
                # pairwise term grow with the square of the repeat count.
                unique_mappings, detection_counts, seen_keys = [], {}, set()
                for item in mappings:
                    item_key = (str(item["pii_type"]).lower(), simple_norm(item["pii_text"]))
                    detection_counts[item_key] = detection_counts.get(item_key, 0) + 1
                    if item_key[1] and item_key in seen_keys:
                        continue
                    seen_keys.add(item_key)
                    unique_mappings.append(item)

                findings = [
                    {
                        "pii_type": item["pii_type"],
                        "pii_text": item["pii_text"],
                        "count": max(
                            detection_counts[(str(item["pii_type"]).lower(), simple_norm(item["pii_text"]))],
                            1 + repeat_counts.get((str(item["pii_type"]).lower(), simple_norm(item["pii_text"])), 0),
                        ),
                    }
                    for item in unique_mappings
                ]

                risk = lcprs.calculate_lcprs(
                    unique_mappings,
                    width,
                    height,
                )
                # Keep the current UI keys, but source the score from the same
                # LCPRS v3 implementation as process_new_document.py.
                risk["normalized_lcprs_score"] = risk["score"]
                risk["number_of_usable_pii"] = len(unique_mappings)
                risk["repeated_values"] = len(mappings) - len(unique_mappings)
                risk["number_of_excluded_pii"] = len(unmapped_entities)
                risk["unmapped_detections"] = unmapped_entities
                risk["detected_entity_count"] = len(detected)

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
                        "findings": findings,
                        "regions": regions,
                        "document_elements": document_elements,
                        "extra_mask_boxes": extra_mask_boxes,
                        "qr_count": len(qr_boxes),
                        "unmapped_entities": unmapped_entities,
                    }
                )

            except Exception as error:

                progress.empty()

                st.error(
                    f"Page {page_number} analysis failed."
                )

                st.exception(error)


    if not results:
        st.error("No pages could be analyzed.")
        st.stop()

    outputs = build_outputs(results, pages, config, uploaded_file.name)
    st.session_state.analysis = {
        "file_key": file_key,
        "file_name": uploaded_file.name,
        "results": results,
        **outputs,
    }
    # Re-run once so transient progress widgets disappear and results render
    # from session state (downloads then never wipe the page).
    st.rerun()

analysis = st.session_state.get("analysis")
if analysis and analysis.get("file_key") == file_key:
    render_results(analysis)


# ============================================================
# FOOTER
# ============================================================

md(
    """
<div class="footer-note">
Layout-Contextual Privacy Risk Assessment<br>
Automated detection can miss or misclassify content. Review every page of any masked output before sharing externally.
</div>
""",
    unsafe_allow_html=True,
)