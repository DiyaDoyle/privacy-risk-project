from __future__ import annotations

import json
import math
import re
import sys
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR, LayoutDetection
from gliner2 import GLiNER2


# ============================================================
# PHASE 2 - ACTUAL MODEL PIPELINE
#
# Phase 1 input:
#   evaluation/dataset/images/*.png
#   evaluation/dataset/annotations/*.json
#
# Models:
#   PaddleOCR
#   GLiNER2-PII
#   PP-DocLayout-L
#
# Then:
#   PII -> OCR bbox -> layout region -> LCPRS
#
# Existing pipeline.py, stage1_mapping.py, layout_mapping.py
# and lcprs_v2.py are NOT modified by this adapter.
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "evaluation" / "dataset"
IMAGES = DATASET / "images"
ANNOTATIONS = DATASET / "annotations"

OUT = ROOT / "evaluation" / "phase2_results"
STAGE1 = OUT / "stage1"
STAGE2 = OUT / "stage2"
LCPRS = OUT / "lcprs"
FINAL = OUT / "final"
AUDIT = OUT / "audit"

LABEL_FILE = ROOT / "config" / "pii_labels.json"
CONFIG_FILE = ROOT / "config" / "lcprs_config.json"

GLINER_MODEL = "fastino/gliner2-privacy-filter-PII-multi"
LAYOUT_MODEL = "PP-DocLayout-L"
GLINER_THRESHOLD = 0.5


# ============================================================
# BASIC HELPERS
# ============================================================

def die(message):
    print("\n" + "=" * 70)
    print("PHASE 2 FAILED")
    print("=" * 70)
    print(message)
    print("=" * 70)
    raise SystemExit(1)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def norm(text):
    return re.sub(r"[^a-zA-Z0-9]", "", str(text)).lower()


def iou(a, b):
    if not a or not b:
        return 0.0

    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ab = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aa + ab - inter

    return inter / union if union > 0 else 0.0


def center(box):
    x1, y1, x2, y2 = map(float, box)
    return (x1 + x2) / 2, (y1 + y2) / 2


def inside(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def union_box(words):
    if not words:
        return None

    return [
        min(w["bbox"][0] for w in words),
        min(w["bbox"][1] for w in words),
        max(w["bbox"][2] for w in words),
        max(w["bbox"][3] for w in words),
    ]


# ============================================================
# CONFIGURATION
# ============================================================

def load_configuration():
    if not LABEL_FILE.exists():
        die(f"Missing PII labels:\n{LABEL_FILE}")

    if not CONFIG_FILE.exists():
        die(f"Missing LCPRS configuration:\n{CONFIG_FILE}")

    labels = load_json(LABEL_FILE)
    config = load_json(CONFIG_FILE)

    if not isinstance(labels, list) or not labels:
        die("config/pii_labels.json must be a non-empty JSON list.")

    # Current lcprs_v2 uses "formula".
    # Support the older "weights" shape as a compatibility fallback.
    if "formula" in config:
        f = config["formula"]
        region_weight = float(f["region_weight"])
        distance_weight = float(f["distance_weight"])
        distance_function = f.get("distance_function", "exponential")
        decay = float(f.get("distance_decay_ratio", 0.15))
    elif "weights" in config:
        f = config["weights"]
        region_weight = float(f["region"])
        distance_weight = float(f["distance"])
        distance_function = "linear"
        decay = 0.15
    else:
        die("lcprs_config.json has neither 'formula' nor 'weights'.")

    severity = config.get("severity", {})
    thresholds = config.get("risk_thresholds", {})
    actions = config.get("governance_actions", {
        "Low": "Normal handling; standard document controls.",
        "Medium": "Apply controlled access and review before sharing.",
        "High": "Restrict access and review before external sharing.",
        "Critical": (
            "Restrict access, protect sensitive content, "
            "and require authorization before sharing."
        ),
    })

    if region_weight < 0 or distance_weight < 0:
        die("LCPRS weights cannot be negative.")

    if region_weight + distance_weight <= 0:
        die("At least one LCPRS weight must be greater than zero.")

    if distance_function not in ("linear", "exponential"):
        die(f"Unsupported LCPRS distance function: {distance_function}")

    if decay <= 0:
        die("LCPRS distance_decay_ratio must be greater than zero.")

    for key in ("low", "medium", "high"):
        if key not in thresholds:
            die(f"Missing LCPRS threshold: {key}")

    for key in ("Low", "Medium", "High", "Critical"):
        if key not in actions:
            die(f"Missing governance action: {key}")

    return {
        "labels": [str(x) for x in labels if str(x).strip()],
        "region_weight": region_weight,
        "distance_weight": distance_weight,
        "distance_function": distance_function,
        "decay": decay,
        "severity": severity,
        "thresholds": thresholds,
        "actions": actions,
    }


# ============================================================
# GLINER
# ============================================================

def flatten_gliner(result):
    entities = []

    if not isinstance(result, dict):
        return entities

    groups = result.get("entities", {})
    if not isinstance(groups, dict):
        return entities

    for label, values in groups.items():
        if not isinstance(values, list):
            continue

        for value in values:
            if isinstance(value, str):
                entities.append({
                    "text": value,
                    "type": str(label),
                    "score": 0.0,
                })
                continue

            if not isinstance(value, dict):
                continue

            text = (
                value.get("text")
                or value.get("entity")
                or value.get("value")
            )
            if not text:
                continue

            score = value.get(
                "score",
                value.get("confidence", 0.0),
            )

            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0

            entities.append({
                "text": str(text),
                "type": str(label),
                "score": score,
            })

    return entities


# ============================================================
# OCR
# ============================================================

def ocr_words(result):
    """
    Extract PaddleOCR word-level text and bounding boxes.

    PaddleOCR returns:
        text_word       -> list of lines, each containing word fragments
        text_word_boxes -> list of numpy arrays, one array per line

    IMPORTANT:
    PaddleOCR's text_word list contains spaces and punctuation as
    separate elements. We ignore those fragments for coordinate
    matching but preserve the actual text fragments.
    """

    rec_texts = result.get("rec_texts", [])
    text_words = result.get("text_word", [])
    text_word_boxes = result.get("text_word_boxes", [])

    words = []

    for line_no, line_words in enumerate(text_words):

        if line_no >= len(text_word_boxes):
            break

        line_boxes = text_word_boxes[line_no]

        if not isinstance(line_words, (list, tuple)):
            line_words = [line_words]

        for word_index, (text, box) in enumerate(
            zip(line_words, line_boxes)
        ):

            text = str(text)

            # Ignore whitespace / punctuation-only fragments.
            if not norm(text):
                continue

            try:
                if hasattr(box, "tolist"):
                    box = box.tolist()

                box = [float(v) for v in box]

            except Exception:
                continue

            if len(box) != 4:
                continue

            words.append({
                "text": text,
                "bbox": box,
                "line": line_no,
                "word_index": word_index,
            })

    combined_text = "\n".join(
        str(x) for x in rec_texts if str(x).strip()
    )

    return combined_text, words


def find_words(entity_text, words):
    target = norm(entity_text)
    if not target:
        return []

    # Exact single-word match.
    for word in words:
        if norm(word["text"]) == target:
            return [word]

    lines = {}
    for word in words:
        lines.setdefault(word["line"], []).append(word)

    # Exact consecutive OCR words.
    for line_words in lines.values():
        for start in range(len(line_words)):
            combined = ""
            selected = []

            for end in range(
                start,
                min(len(line_words), start + 30),
            ):
                current = norm(line_words[end]["text"])
                if not current:
                    continue

                combined += current
                selected.append(line_words[end])

                if combined == target:
                    return selected

                if len(combined) > len(target):
                    break

    # Strong fuzzy match.
    best = []
    best_score = 0.0

    for line_words in lines.values():
        for start in range(len(line_words)):
            combined = ""
            selected = []

            for end in range(
                start,
                min(len(line_words), start + 30),
            ):
                current = norm(line_words[end]["text"])
                if not current:
                    continue

                combined += current
                selected.append(line_words[end])

                if len(combined) >= max(1, int(len(target) * 0.75)):
                    score = SequenceMatcher(
                        None, target, combined
                    ).ratio()

                    if score > best_score:
                        best_score = score
                        best = list(selected)

                if len(combined) > len(target) + 20:
                    break

    return best if best_score >= 0.88 else []


# ============================================================
# LAYOUT
# ============================================================

def extract_layout(result):
    data = result.json
    if isinstance(data, str):
        data = json.loads(data)

    if not isinstance(data, dict):
        return []

    data = data.get("res", data)
    boxes = data.get("boxes", [])

    regions = []

    for idx, item in enumerate(boxes):
        if not isinstance(item, dict):
            continue

        coordinate = item.get("coordinate")
        if coordinate is None:
            coordinate = item.get("bbox")

        if coordinate is None:
            continue

        try:
            coordinate = [float(v) for v in coordinate]
        except Exception:
            continue

        if len(coordinate) != 4:
            continue

        x1, y1, x2, y2 = coordinate
        if x2 <= x1 or y2 <= y1:
            continue

        regions.append({
            "region_id": idx,
            "label": str(item.get("label", "unknown")),
            "score": float(item.get("score", 0.0)),
            "bbox": coordinate,
        })

    return regions


def map_to_layout(pii_box, regions):
    if not pii_box:
        return None, "unmatched"

    p = center(pii_box)

    containing = [
        r for r in regions
        if inside(p, r["bbox"])
    ]

    if containing:
        containing.sort(
            key=lambda r:
            (r["bbox"][2] - r["bbox"][0])
            * (r["bbox"][3] - r["bbox"][1])
        )
        return containing[0], "center_inside"

    best = None
    best_iou = 0.0

    for r in regions:
        score = iou(pii_box, r["bbox"])
        if score > best_iou:
            best_iou = score
            best = r

    if best is not None and best_iou > 0:
        return best, "highest_iou"

    return None, "unmatched"


# ============================================================
# LCPRS
# ============================================================

def severity_for(label, config):
    try:
        value = float(config["severity"][label])
    except (KeyError, TypeError, ValueError):
        return None

    return value if value >= 0 else None


def distance_closeness(a, b, width, height, config):
    ax, ay = center(a)
    bx, by = center(b)

    distance = math.hypot(ax - bx, ay - by)
    diagonal = math.hypot(width, height)

    if diagonal <= 0:
        return 0.0

    d = distance / diagonal

    if config["distance_function"] == "exponential":
        value = math.exp(-d / config["decay"])
    else:
        value = 1.0 - d

    return max(0.0, min(1.0, value))


def run_lcprs(items, width, height, config):
    usable = []
    excluded = []

    for item in items:
        severity = severity_for(item["pii_type"], config)

        if severity is None:
            excluded.append({
                "pii_text": item["pii_text"],
                "pii_type": item["pii_type"],
                "reason": "No configured LCPRS severity.",
            })
        else:
            usable.append(item)

    standalone = sum(
        severity_for(x["pii_type"], config)
        for x in usable
    )

    maximum_relationship = (
        config["region_weight"]
        + config["distance_weight"]
    )

    pairs = []

    for a, b in combinations(usable, 2):
        sa = severity_for(a["pii_type"], config)
        sb = severity_for(b["pii_type"], config)

        same = (
            a.get("layout_region_id") is not None
            and b.get("layout_region_id") is not None
            and a["layout_region_id"] == b["layout_region_id"]
        )

        close = distance_closeness(
            a["pii_bbox"],
            b["pii_bbox"],
            width,
            height,
            config,
        )

        relationship = (
            config["region_weight"] * (1.0 if same else 0.0)
            + config["distance_weight"] * close
        )

        pair_risk = (sa + sb) * relationship

        pairs.append({
            "pii_a": a["pii_text"],
            "pii_a_type": a["pii_type"],
            "severity_a": sa,
            "pii_b": b["pii_text"],
            "pii_b_type": b["pii_type"],
            "severity_b": sb,
            "same_region": same,
            "distance_closeness": close,
            "relationship": relationship,
            "pair_risk": pair_risk,
        })

    pair_total = sum(x["pair_risk"] for x in pairs)
    document_risk = standalone + pair_total

    maximum_pair = 0.0

    for a, b in combinations(usable, 2):
        sa = severity_for(a["pii_type"], config)
        sb = severity_for(b["pii_type"], config)
        maximum_pair += (sa + sb) * maximum_relationship

    maximum = standalone + maximum_pair

    score = (
        document_risk / maximum * 100.0
        if maximum > 0 else 0.0
    )

    score = max(0.0, min(100.0, score))

    thresholds = config["thresholds"]

    if score < float(thresholds["low"]):
        category = "Low"
    elif score < float(thresholds["medium"]):
        category = "Medium"
    elif score < float(thresholds["high"]):
        category = "High"
    else:
        category = "Critical"

    return {
        "number_of_usable_pii": len(usable),
        "number_of_excluded_pii": len(excluded),
        "excluded_pii": excluded,
        "standalone_risk": standalone,
        "pair_risk_total": pair_total,
        "document_risk": document_risk,
        "maximum_possible_risk": maximum,
        "normalized_lcprs_score": score,
        "risk_category": category,
        "governance_action": config["actions"][category],
        "pii_items_used": usable,
        "pair_results": pairs,
    }


# ============================================================
# VISUALIZATIONS
# ============================================================

def font(size=14):
    path = Path(r"C:\Windows\Fonts\arial.ttf")
    return (
        ImageFont.truetype(str(path), size)
        if path.exists()
        else ImageFont.load_default()
    )


def stage1_image(path, comparisons, output):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    f = font(12)

    for item in comparisons:
        gt = item["ground_truth_bbox"]
        pred = item["predicted_bbox"]

        if gt:
            draw.rectangle(
                [int(v) for v in gt],
                outline="green",
                width=2,
            )

        if pred:
            draw.rectangle(
                [int(v) for v in pred],
                outline="red",
                width=2,
            )

    image.save(output, format="PNG")


def stage2_image(path, regions, mappings, output):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    rf = font(11)

    for region in regions:
        draw.rectangle(
            [int(v) for v in region["bbox"]],
            outline="blue",
            width=2,
        )
        x1, y1, _, _ = region["bbox"]
        draw.text(
            (int(x1), max(0, int(y1) - 13)),
            f'{region["region_id"]}:{region["label"]}',
            fill="blue",
            font=rf,
        )

    for item in mappings:
        box = item.get("pii_bbox")
        if not box:
            continue

        color = (
            "red"
            if item.get("layout_region_id") is not None
            else "orange"
        )

        draw.rectangle(
            [int(v) for v in box],
            outline=color,
            width=3,
        )

    image.save(output, format="PNG")

# ============================================================
# FINAL UNIFIED VISUALIZATION
# ============================================================

def final_image(
    image_path,
    mappings,
    risk,
    output,
):
    """
    Create one clean user-facing visualization.

    PII types are NOT hardcoded.

    Only PII items supplied by LCPRS are visualized.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    shown = set()

    # --------------------------------------------------------
    # Draw PII items supplied by LCPRS
    # --------------------------------------------------------

    for item in mappings:

        pii_type = str(
            item.get("pii_type", "")
        ).strip()

        pii_text = str(
            item.get("pii_text", "")
        ).strip()

        box = item.get("pii_bbox")

        if not pii_type or not pii_text or not box:
            continue

        try:
            key = (
                pii_type.lower(),
                pii_text.lower(),
                tuple(
                    round(float(v), 1)
                    for v in box
                ),
            )
        except Exception:
            continue

        if key in shown:
            continue

        shown.add(key)

        try:
            x1, y1, x2, y2 = [
                int(float(v))
                for v in box
            ]
        except Exception:
            continue

        width, height = image.size

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        # PII bounding box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3,
        )

        # Dynamic PII type label
        label = pii_type.replace(
            "_",
            " "
        ).upper()

        label_font = font(11)

        text_box = draw.textbbox(
            (0, 0),
            label,
            font=label_font,
        )

        label_width = (
            text_box[2]
            - text_box[0]
            + 8
        )

        label_height = (
            text_box[3]
            - text_box[1]
            + 5
        )

        label_y = max(
            0,
            y1 - label_height
        )

        draw.rectangle(
            [
                x1,
                label_y,
                x1 + label_width,
                label_y + label_height,
            ],
            fill="white",
        )

        draw.text(
            (
                x1 + 4,
                label_y + 2,
            ),
            label,
            fill="red",
            font=label_font,
        )

    # ========================================================
    # RESULT PANEL
    # ========================================================

    panel_width = 280
    width, height = image.size

    canvas = Image.new(
        "RGB",
        (
            width + panel_width,
            height,
        ),
        "white",
    )

    canvas.paste(
        image,
        (0, 0),
    )

    panel = ImageDraw.Draw(canvas)

    x = width + 20
    y = 30

    panel.text(
        (x, y),
        "PRIVACY RISK",
        fill="black",
        font=font(21),
    )

    y += 50

    panel.text(
        (x, y),
        "LCPRS SCORE",
        fill="black",
        font=font(13),
    )

    y += 27

    panel.text(
        (x, y),
        f"{float(risk['normalized_lcprs_score']):.2f} / 100",
        fill="black",
        font=font(23),
    )

    y += 55

    panel.text(
        (x, y),
        "RISK",
        fill="black",
        font=font(13),
    )

    y += 27

    panel.text(
        (x, y),
        str(risk["risk_category"]).upper(),
        fill="black",
        font=font(23),
    )

    y += 55

    panel.text(
        (x, y),
        "PII DETECTED",
        fill="black",
        font=font(13),
    )

    y += 27

    panel.text(
        (x, y),
        str(risk["number_of_usable_pii"]),
        fill="black",
        font=font(23),
    )

    y += 55

    panel.text(
        (x, y),
        "GOVERNANCE",
        fill="black",
        font=font(13),
    )

    y += 27

    action = str(
        risk.get(
            "governance_action",
            "",
        )
    )

    words = action.split()
    line = ""

    for word in words:

        test = (
            line + " " + word
        ).strip()

        if len(test) > 28:

            if line:
                panel.text(
                    (x, y),
                    line,
                    fill="black",
                    font=font(11),
                )
                y += 19

            line = word

        else:
            line = test

    if line:
        panel.text(
            (x, y),
            line,
            fill="black",
            font=font(11),
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output,
        format="PNG",
    )

    print(
        f"Final visualization: {output}"
    )


# ============================================================
# FINAL-ONLY MODE
# ============================================================

def generate_final_only(images):
    """
    Generate final visualizations from existing
    Phase 2 LCPRS JSON results.

    Does not load or execute any ML models.
    """

    FINAL.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "GENERATING FINAL VISUALIZATIONS"
    )
    print(
        "USING EXISTING PHASE 2 RESULTS"
    )
    print(
        "=" * 70
    )

    generated = 0
    skipped = 0

    for image_path in images:

        document = image_path.stem

        lcprs_path = (
            LCPRS
            /
            f"{document}_lcprs.json"
        )

        if not lcprs_path.exists():

            print(
                f"SKIPPED {document}: "
                "LCPRS result missing."
            )

            skipped += 1
            continue

        lcprs = load_json(
            lcprs_path
        )

        # Source of truth: existing LCPRS output.
        mappings = lcprs.get(
            "pii_items_used",
            []
        )

        risk = {
            "normalized_lcprs_score":
                lcprs.get(
                    "normalized_lcprs_score",
                    0.0,
                ),

            "risk_category":
                lcprs.get(
                    "risk_category",
                    "Unknown",
                ),

            "number_of_usable_pii":
                lcprs.get(
                    "number_of_usable_pii",
                    len(mappings),
                ),

            "governance_action":
                lcprs.get(
                    "governance_action",
                    "",
                ),
        }

        output = (
            FINAL
            /
            f"{document}_final.png"
        )

        final_image(
            image_path,
            mappings,
            risk,
            output,
        )

        generated += 1

    print(
        "\n" + "=" * 70
    )
    print(
        "FINAL VISUALIZATION GENERATION COMPLETE"
    )
    print(
        "=" * 70
    )

    print(
        f"Generated : {generated}"
    )

    print(
        f"Skipped   : {skipped}"
    )

    print(
        f"Output    : {FINAL}"
    )



# ============================================================
# PHASE 2 AUDIT
# ============================================================

def _close(a, b, tolerance=1e-6):
    try:
        return abs(float(a) - float(b)) <= tolerance
    except (TypeError, ValueError):
        return False


def audit_document(document, config):
    """
    Audit an existing LCPRS JSON file without rerunning any ML model.

    The audit independently reconstructs the stored LCPRS arithmetic from:
        pii_items_used
        pair_results
        formula weights
        configured thresholds

    No PII labels or risk categories are hardcoded here.
    """
    path = LCPRS / f"{document}_lcprs.json"

    if not path.exists():
        return {
            "document": document,
            "status": "missing",
            "arithmetic_consistent": False,
            "error": f"Missing LCPRS result: {path}",
        }

    data = load_json(path)
    items = data.get("pii_items_used", [])
    pairs = data.get("pair_results", [])

    region_weight = float(
        data.get("formula", {}).get(
            "region_weight",
            config["region_weight"],
        )
    )
    distance_weight = float(
        data.get("formula", {}).get(
            "distance_weight",
            config["distance_weight"],
        )
    )

    stored_standalone = float(data.get("standalone_risk", 0.0))
    stored_pair_total = float(data.get("pair_risk_total", 0.0))
    stored_document_risk = float(data.get("document_risk", 0.0))
    stored_maximum = float(data.get("maximum_possible_risk", 0.0))
    stored_score = float(data.get("normalized_lcprs_score", 0.0))

    calculated_standalone = 0.0
    missing_severity = []

    for item in items:
        severity = severity_for(item.get("pii_type", ""), config)
        if severity is None:
            missing_severity.append(item.get("pii_type", ""))
        else:
            calculated_standalone += severity

    calculated_pair_total = sum(
        float(pair.get("pair_risk", 0.0))
        for pair in pairs
    )

    calculated_document_risk = (
        calculated_standalone + calculated_pair_total
    )

    maximum_relationship = region_weight + distance_weight
    calculated_maximum = 0.0

    for a, b in combinations(items, 2):
        sa = severity_for(a.get("pii_type", ""), config)
        sb = severity_for(b.get("pii_type", ""), config)

        if sa is None or sb is None:
            continue

        calculated_maximum += (
            (sa + sb) * maximum_relationship
        )

    calculated_maximum += calculated_standalone

    calculated_score = (
        calculated_document_risk
        / calculated_maximum
        * 100.0
        if calculated_maximum > 0
        else 0.0
    )

    calculated_score = max(
        0.0,
        min(100.0, calculated_score),
    )

    thresholds = config["thresholds"]

    if calculated_score < float(thresholds["low"]):
        calculated_category = "Low"
    elif calculated_score < float(thresholds["medium"]):
        calculated_category = "Medium"
    elif calculated_score < float(thresholds["high"]):
        calculated_category = "High"
    else:
        calculated_category = "Critical"

    checks = {
        "standalone_risk": _close(
            stored_standalone,
            calculated_standalone,
        ),
        "pair_risk_total": _close(
            stored_pair_total,
            calculated_pair_total,
        ),
        "document_risk": _close(
            stored_document_risk,
            calculated_document_risk,
        ),
        "maximum_possible_risk": _close(
            stored_maximum,
            calculated_maximum,
        ),
        "normalized_score": _close(
            stored_score,
            calculated_score,
        ),
        "risk_category": (
            str(data.get("risk_category", "")).lower()
            == calculated_category.lower()
        ),
    }

    return {
        "document": document,
        "status": "audited",
        "arithmetic_consistent": all(checks.values())
        and not missing_severity,
        "checks": checks,
        "stored": {
            "standalone_risk": stored_standalone,
            "pair_risk_total": stored_pair_total,
            "document_risk": stored_document_risk,
            "maximum_possible_risk": stored_maximum,
            "normalized_lcprs_score": stored_score,
            "risk_category": data.get("risk_category"),
        },
        "recalculated": {
            "standalone_risk": calculated_standalone,
            "pair_risk_total": calculated_pair_total,
            "document_risk": calculated_document_risk,
            "maximum_possible_risk": calculated_maximum,
            "normalized_lcprs_score": calculated_score,
            "risk_category": calculated_category,
        },
        "number_of_usable_pii": len(items),
        "layout_regions_used": len({
            item.get("layout_region_id")
            for item in items
            if item.get("layout_region_id") is not None
        }),
        "pair_count": len(pairs),
        "missing_severity_labels": missing_severity,
    }


def _annotation_group_info(image_path):
    """
    Derive the experimental condition and base group from the
    annotation metadata. No condition names are hardcoded.
    """
    annotation_path = ANNOTATIONS / f"{image_path.stem}.json"

    if not annotation_path.exists():
        return None, image_path.stem

    annotation = load_json(annotation_path)
    condition = str(annotation.get("condition", "")).strip()

    base_group = str(
        annotation.get("base_page_id")
        or annotation.get("base_image_id")
        or annotation.get("source_page_id")
        or annotation.get("source_image_id")
        or annotation.get("source")
        or ""
    ).strip()

    if not base_group:
        base_group = image_path.stem

    if condition and image_path.stem.endswith(
        "_" + condition
    ):
        base_group = image_path.stem[
            :-(len(condition) + 1)
        ]

    return condition or "unspecified", base_group


def create_audit_report(audit_rows, paired_rows):
    """
    Save a machine-readable JSON audit and a human-readable PNG report.
    """
    AUDIT.mkdir(parents=True, exist_ok=True)

    arithmetic_ok = sum(
        bool(row.get("arithmetic_consistent"))
        for row in audit_rows
    )

    report = {
        "phase": 2,
        "audit_type": "LCPRS arithmetic and layout audit",
        "documents_audited": len(audit_rows),
        "arithmetic_consistent_documents": arithmetic_ok,
        "documents": audit_rows,
        "paired_layout_variants": paired_rows,
    }

    save_json(
        AUDIT / "phase2_audit.json",
        report,
    )

    # --------------------------------------------------------
    # Human-readable audit image
    # --------------------------------------------------------

    title_font = font(22)
    header_font = font(12)
    body_font = font(11)

    row_height = 24
    margin = 25

    columns = [
        ("DOCUMENT", 260),
        ("CONDITION", 115),
        ("LCPRS", 75),
        ("RISK", 85),
        ("PII", 55),
        ("REGIONS", 75),
        ("ARITHMETIC", 100),
    ]

    table_width = sum(width for _, width in columns)
    table_height = (
        95
        + (len(audit_rows) + 1) * row_height
        + 80
        + max(1, len(paired_rows)) * row_height
    )

    report_image = Image.new(
        "RGB",
        (
            table_width + margin * 2,
            table_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(report_image)

    y = margin
    draw.text(
        (margin, y),
        "PHASE 2 — LCPRS AUDIT",
        fill="black",
        font=title_font,
    )

    y += 35

    draw.text(
        (margin, y),
        (
            f"Arithmetic consistent: "
            f"{arithmetic_ok}/{len(audit_rows)} documents"
        ),
        fill="black",
        font=header_font,
    )

    y += 30

    x = margin

    for heading, width in columns:
        draw.rectangle(
            [x, y, x + width, y + row_height],
            outline="black",
            width=1,
        )
        draw.text(
            (x + 4, y + 5),
            heading,
            fill="black",
            font=header_font,
        )
        x += width

    y += row_height

    for row in audit_rows:
        x = margin

        values = [
            row.get("document", ""),
            row.get("condition", ""),
            f'{row.get("score", 0.0):.2f}',
            str(row.get("risk", "")),
            str(row.get("pii", "")),
            str(row.get("regions", "")),
            (
                "PASS"
                if row.get("arithmetic_consistent")
                else "CHECK"
            ),
        ]

        for (_, width), value in zip(columns, values):
            draw.rectangle(
                [x, y, x + width, y + row_height],
                outline="black",
                width=1,
            )
            draw.text(
                (x + 4, y + 5),
                str(value)[:38],
                fill="black",
                font=body_font,
            )
            x += width

        y += row_height

    y += 25

    draw.text(
        (margin, y),
        "PAIRED LAYOUT VARIANTS",
        fill="black",
        font=header_font,
    )

    y += 22

    pair_columns = [
        ("BASE GROUP", 300),
        ("CONDITIONS / SCORES", 520),
    ]

    x = margin

    for heading, width in pair_columns:
        draw.rectangle(
            [x, y, x + width, y + row_height],
            outline="black",
            width=1,
        )
        draw.text(
            (x + 4, y + 5),
            heading,
            fill="black",
            font=header_font,
        )
        x += width

    y += row_height

    for pair in paired_rows:
        x = margin

        condition_text = " | ".join(
            f'{item["condition"]}: '
            f'{item["score"]:.2f} '
            f'({item["risk"]})'
            for item in pair["variants"]
        )

        values = [
            pair["base_group"],
            condition_text,
        ]

        for (_, width), value in zip(
            pair_columns,
            values,
        ):
            draw.rectangle(
                [x, y, x + width, y + row_height],
                outline="black",
                width=1,
            )
            draw.text(
                (x + 4, y + 5),
                str(value)[:85],
                fill="black",
                font=body_font,
            )
            x += width

        y += row_height

    output = AUDIT / "phase2_audit.png"

    report_image.save(
        output,
        format="PNG",
    )

    print(f"\nAudit JSON: {AUDIT / 'phase2_audit.json'}")
    print(f"Audit report: {output}")


def run_audit(images, config):
    """
    Audit existing results only. No OCR, GLiNER2 or PP-DocLayout run.
    """
    print("\n" + "=" * 70)
    print("PHASE 2 AUDIT")
    print("=" * 70)

    audit_rows = []
    grouped = {}

    for image_path in images:
        document = image_path.stem
        condition, base_group = _annotation_group_info(
            image_path
        )

        result = audit_document(
            document,
            config,
        )

        lcprs_path = LCPRS / f"{document}_lcprs.json"

        if lcprs_path.exists():
            data = load_json(lcprs_path)

            row = {
                "document": document,
                "condition": condition,
                "score": float(
                    data.get(
                        "normalized_lcprs_score",
                        0.0,
                    )
                ),
                "risk": data.get(
                    "risk_category",
                    "Unknown",
                ),
                "pii": int(
                    data.get(
                        "number_of_usable_pii",
                        len(data.get("pii_items_used", [])),
                    )
                ),
                "regions": len({
                    item.get("layout_region_id")
                    for item in data.get(
                        "pii_items_used",
                        [],
                    )
                    if item.get("layout_region_id") is not None
                }),
                "arithmetic_consistent": bool(
                    result.get(
                        "arithmetic_consistent",
                        False,
                    )
                ),
            }

            grouped.setdefault(
                base_group,
                [],
            ).append(row)

        else:
            row = {
                "document": document,
                "condition": condition,
                "score": 0.0,
                "risk": "Missing",
                "pii": 0,
                "regions": 0,
                "arithmetic_consistent": False,
            }

        audit_rows.append({
            **result,
            "condition": condition,
            "base_group": base_group,
            "score": row["score"],
            "risk": row["risk"],
            "pii": row["pii"],
            "regions": row["regions"],
        })

    paired_rows = []

    for base_group, variants in grouped.items():
        if len(variants) < 2:
            continue

        paired_rows.append({
            "base_group": base_group,
            "variants": sorted(
                variants,
                key=lambda x: x["condition"],
            ),
        })

    print(
        f"Documents audited : {len(audit_rows)}"
    )

    print(
        "Arithmetic consistent : "
        f"{sum(bool(x.get('arithmetic_consistent')) for x in audit_rows)}"
        f"/{len(audit_rows)}"
    )

    print("\nPaired layout variants:")

    for pair in paired_rows:
        print(f"\n{pair['base_group']}")

        for variant in pair["variants"]:
            print(
                f"  {variant['condition']}: "
                f"{variant['score']:.2f} "
                f"({variant['risk']}) | "
                f"PII={variant['pii']} | "
                f"regions={variant['regions']}"
            )

    create_audit_report(
        audit_rows,
        paired_rows,
    )

# ============================================================
# ONE DOCUMENT
# ============================================================

def process(
    image_path,
    annotation_path,
    ocr,
    gliner,
    layout_model,
    config,
):
    document = image_path.stem
    gt = load_json(annotation_path)
    gt_entities = gt.get("entities", [])

    with Image.open(image_path) as image:
        width, height = image.size

    print("\n" + "=" * 70)
    print(f"PROCESSING {document}")
    print("=" * 70)
    print(f"Ground-truth PII: {len(gt_entities)}")

    # --------------------------------------------------------
    # PaddleOCR
    # --------------------------------------------------------

    print("Running PaddleOCR...")

    ocr_results = list(
        ocr.predict(input=str(image_path))
    )

    if not ocr_results:
        raise RuntimeError("PaddleOCR returned no result.")

    text, words = ocr_words(ocr_results[0])

    if not text.strip():
        raise RuntimeError("PaddleOCR extracted no text.")

    coordinate_words = [
        x for x in words if x.get("bbox") is not None
    ]

    print(f"OCR word boxes: {len(coordinate_words)}")

    # --------------------------------------------------------
    # GLiNER2
    # --------------------------------------------------------

    print("Running GLiNER2-PII...")

    result = gliner.extract_entities(
        text,
        config["labels"],
        threshold=GLINER_THRESHOLD,
        include_confidence=True,
        include_spans=True,
    )

    detected = flatten_gliner(result)

    print(f"GLiNER entities detected: {len(detected)}")

    # --------------------------------------------------------
    # GLiNER -> OCR bbox
    # --------------------------------------------------------

    predictions = []

    for entity in detected:
        matched = find_words(
            entity["text"],
            coordinate_words,
        )

        box = union_box(matched)

        if box is None:
            continue

        predictions.append({
            "text": entity["text"],
            "type": entity["type"],
            "score": entity["score"],
            "ocr_words": [
                x["text"] for x in matched
            ],
            "ocr_word_boxes": [
                x["bbox"] for x in matched
            ],
            "predicted_bbox": box,
        })

    print(f"PII mapped to OCR: {len(predictions)}")

    # --------------------------------------------------------
    # Match to synthetic ground truth
    # --------------------------------------------------------

    comparisons = []
    used = set()

    for gt_item in gt_entities:
        gt_text = str(gt_item.get("text", ""))
        gt_type = str(gt_item.get("type", ""))
        gt_box = gt_item.get("bbox")

        found = None
        found_idx = None

        # Exact text + type.
        for idx, pred in enumerate(predictions):
            if idx in used:
                continue

            if (
                norm(gt_text) == norm(pred["text"])
                and gt_type == pred["type"]
            ):
                found = pred
                found_idx = idx
                break

        # Exact text fallback.
        if found is None:
            for idx, pred in enumerate(predictions):
                if idx in used:
                    continue

                if norm(gt_text) == norm(pred["text"]):
                    found = pred
                    found_idx = idx
                    break

        if found is None:
            comparisons.append({
                "ground_truth_text": gt_text,
                "ground_truth_type": gt_type,
                "ground_truth_bbox": gt_box,
                "predicted_text": None,
                "predicted_type": None,
                "predicted_bbox": None,
                "iou": 0.0,
            })
            continue

        used.add(found_idx)

        comparisons.append({
            "ground_truth_text": gt_text,
            "ground_truth_type": gt_type,
            "ground_truth_bbox": gt_box,
            "predicted_text": found["text"],
            "predicted_type": found["type"],
            "predicted_confidence": found["score"],
            "predicted_bbox": found["predicted_bbox"],
            "iou": iou(
                gt_box,
                found["predicted_bbox"],
            ),
        })

    matched = [
        x for x in comparisons
        if x["predicted_bbox"] is not None
    ]

    average_iou = (
        sum(x["iou"] for x in matched) / len(matched)
        if matched else 0.0
    )

    # --------------------------------------------------------
    # Stage 1 output
    # --------------------------------------------------------

    save_json(
        STAGE1 / f"{document}_stage1.json",
        {
            "document": document,
            "ground_truth_count": len(gt_entities),
            "gliner_detected_count": len(detected),
            "ocr_mapped_count": len(predictions),
            "ground_truth_matched_count": len(matched),
            "ground_truth_match_rate": (
                len(matched) / len(gt_entities)
                if gt_entities else 0.0
            ),
            "average_bbox_iou": average_iou,
            "ground_truth_comparisons": comparisons,
            "detected_entities": detected,
            "mapped_predictions": predictions,
        },
    )

    stage1_image(
        image_path,
        comparisons,
        STAGE1 / f"{document}_stage1.png",
    )

    # --------------------------------------------------------
    # PP-DocLayout
    # --------------------------------------------------------

    print("Running PP-DocLayout-L...")

    layout_results = list(
        layout_model.predict(
            input=str(image_path),
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

    print(
        f"Layout regions detected: {len(regions)}"
    )

    # --------------------------------------------------------
    # PII -> layout
    # --------------------------------------------------------

    mappings = []

    for pred in predictions:
        region, method = map_to_layout(
            pred["predicted_bbox"],
            regions,
        )

        mappings.append({
            "pii_text": pred["text"],
            "pii_type": pred["type"],
            "confidence": pred["score"],
            "pii_bbox": pred["predicted_bbox"],
            "ocr_words": pred["ocr_words"],
            "layout_region_id": (
                region["region_id"]
                if region else None
            ),
            "layout_region_type": (
                region["label"]
                if region else None
            ),
            "layout_region_score": (
                region["score"]
                if region else None
            ),
            "layout_region_bbox": (
                region["bbox"]
                if region else None
            ),
            "mapping_method": method,
        })

    mapped_layout = sum(
        x["layout_region_id"] is not None
        for x in mappings
    )

    save_json(
        STAGE2 / f"{document}_layout.json",
        {
            "document": document,
            "image_width": width,
            "image_height": height,
            "layout_region_count": len(regions),
            "pii_prediction_count": len(predictions),
            "pii_layout_mapped_count": mapped_layout,
            "layout_mapping_rate": (
                mapped_layout / len(predictions)
                if predictions else 0.0
            ),
            "layout_regions": regions,
            "pii_mappings": mappings,
        },
    )

    stage2_image(
        image_path,
        regions,
        mappings,
        STAGE2 / f"{document}_layout.png",
    )

    # --------------------------------------------------------
    # LCPRS
    # --------------------------------------------------------

    risk = run_lcprs(
        mappings,
        width,
        height,
        config,
    )

    save_json(
        LCPRS / f"{document}_lcprs.json",
        {
            "lcprs_version": "2.0",
            "document": document,
            "image_width": width,
            "image_height": height,
            "formula": {
                "region_weight": config["region_weight"],
                "distance_weight": config["distance_weight"],
                "distance_function": config["distance_function"],
                "distance_decay_ratio": config["decay"],
            },
            **risk,
        },
    )

    print(
        f"LCPRS usable PII: "
        f"{risk['number_of_usable_pii']}"
    )

    print(
        f"LCPRS score: "
        f"{risk['normalized_lcprs_score']:.2f}/100"
    )

    print(
        f"Risk category: "
        f"{risk['risk_category']}"
    )

    # --------------------------------------------------------
    # FINAL UNIFIED VISUALIZATION
    # --------------------------------------------------------

    final_image(
        image_path,
        mappings,
        risk,
        FINAL / f"{document}_final.png",
    )

    return {
        "document": document,
        "condition": gt.get("condition"),
        "ground_truth_pii": len(gt_entities),
        "gliner_detected": len(detected),
        "ocr_mapped": len(predictions),
        "ground_truth_matched": len(matched),
        "ground_truth_match_rate": (
            len(matched) / len(gt_entities)
            if gt_entities else 0.0
        ),
        "average_bbox_iou": average_iou,
        "layout_regions": len(regions),
        "layout_mapped": mapped_layout,
        "layout_mapping_rate": (
            mapped_layout / len(predictions)
            if predictions else 0.0
        ),
        "lcprs_usable_pii": risk[
            "number_of_usable_pii"
        ],
        "lcprs_score": risk[
            "normalized_lcprs_score"
        ],
        "risk_category": risk[
            "risk_category"
        ],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 2 - ACTUAL MODEL PIPELINE")
    print("=" * 70)

    for directory in (
            OUT,
            STAGE1,
            STAGE2,
            LCPRS,
            FINAL,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    if not IMAGES.exists():
        die(f"Phase 1 images directory not found:\n{IMAGES}")

    if not ANNOTATIONS.exists():
        die(
            f"Phase 1 annotations directory not found:\n"
            f"{ANNOTATIONS}"
        )

    images = sorted(
        IMAGES.glob("*.png")
    )

    if not images:
        die(f"No PNG files found:\n{IMAGES}")

    missing = [
        p.stem
        for p in images
        if not (
            ANNOTATIONS / f"{p.stem}.json"
        ).exists()
    ]

    if missing:
        die(
            "Missing annotations for:\n"
            + "\n".join(missing)
        )

    print(f"Images found: {len(images)}")
    print(f"Annotations found: {len(images)}")

    # --------------------------------------------------------
    # FINAL-ONLY MODE
    #
    # Uses existing Phase 2 results.
    # No ML models are loaded or rerun.
    # --------------------------------------------------------

    config = load_configuration()

    if "--final-only" in sys.argv:
        generate_final_only(images)

        if "--audit" in sys.argv:
            run_audit(images, config)

        return

    if "--audit" in sys.argv:
        run_audit(images, config)
        return


    print(
        f"PII labels loaded: {len(config['labels'])}"
    )

    print(
        f"LCPRS weights: "
        f"region={config['region_weight']}, "
        f"distance={config['distance_weight']}"
    )

    print("\nLoading PaddleOCR...")
    ocr = PaddleOCR(
        lang="en",
        enable_mkldnn=False,
        return_word_box=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    print("PaddleOCR loaded successfully.")

    print("\nLoading GLiNER2-PII...")
    gliner = GLiNER2.from_pretrained(
        GLINER_MODEL
    )
    print("GLiNER2-PII loaded successfully.")

    print("\nLoading PP-DocLayout-L...")
    layout_model = LayoutDetection(
        model_name=LAYOUT_MODEL,
        device="cpu",
        enable_mkldnn=False,
    )
    print("PP-DocLayout-L loaded successfully.")

    summary = []
    errors = []

    for image_path in images:
        annotation_path = (
            ANNOTATIONS / f"{image_path.stem}.json"
        )

        try:
            summary.append(
                process(
                    image_path,
                    annotation_path,
                    ocr,
                    gliner,
                    layout_model,
                    config,
                )
            )

        except Exception as exc:
            print(
                f"\nERROR processing {image_path.name}: "
                f"{type(exc).__name__}: {exc}"
            )

            errors.append({
                "document": image_path.stem,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    total_gt = sum(
        x["ground_truth_pii"]
        for x in summary
    )

    total_detected = sum(
        x["gliner_detected"]
        for x in summary
    )

    total_ocr_mapped = sum(
        x["ocr_mapped"]
        for x in summary
    )

    total_matched = sum(
        x["ground_truth_matched"]
        for x in summary
    )

    total_layout = sum(
        x["layout_mapped"]
        for x in summary
    )

    average_iou = (
        sum(
            x["average_bbox_iou"]
            for x in summary
        ) / len(summary)
        if summary else 0.0
    )

    final = {
        "phase": 2,
        "status": (
            "completed"
            if not errors
            else "failed"
        ),
        "documents_found": len(images),
        "documents_processed": len(summary),
        "documents_with_errors": len(errors),
        "aggregate": {
            "ground_truth_pii": total_gt,
            "gliner_detected": total_detected,
            "ocr_mapped": total_ocr_mapped,
            "ground_truth_matched": total_matched,
            "ground_truth_match_rate": (
                total_matched / total_gt
                if total_gt else 0.0
            ),
            "average_bbox_iou": average_iou,
            "layout_mapped": total_layout,
            "layout_mapping_rate": (
                total_layout / total_ocr_mapped
                if total_ocr_mapped else 0.0
            ),
        },
        "documents": summary,
        "errors": errors,
    }

    save_json(
        OUT / "phase2_summary.json",
        final,
    )

    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)
    print(f"Documents found       : {len(images)}")
    print(f"Documents processed   : {len(summary)}")
    print(f"Documents with errors : {len(errors)}")
    print(f"Ground-truth PII      : {total_gt}")
    print(f"GLiNER detected       : {total_detected}")
    print(f"OCR mapped            : {total_ocr_mapped}")
    print(f"GT matched            : {total_matched}")
    print(
        f"GT match rate         : "
        f"{(total_matched / total_gt * 100) if total_gt else 0:.2f}%"
    )
    print(
        f"Average bbox IoU      : {average_iou:.4f}"
    )
    print(f"Layout mapped         : {total_layout}")
    print(
        f"Layout mapping rate   : "
        f"{(total_layout / total_ocr_mapped * 100) if total_ocr_mapped else 0:.2f}%"
    )

    print(
        f"\nResults:\n{OUT}"
    )

    if errors:
        print(
            "\nPHASE 2 FAILED."
        )
        for error in errors:
            print(
                f'{error["document"]}: '
                f'{error["error_type"]}: '
                f'{error["error"]}'
            )
        raise SystemExit(1)

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

