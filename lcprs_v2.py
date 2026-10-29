
"""
LCPRS v3: Layout-Contextual Privacy Risk Score

Item-level foundation:
    Sensitivity x visibility, motivated by published privacy-scoring work.

Proposed extension:
    Pairwise PII risk using layout-region co-membership and spatial proximity.

Important:
    The complete LCPRS equation is a proposed model, not an equation
    claimed to have been copied from one paper.
    Parameters are read from config/lcprs_config.json.
"""

import json
import math
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "lcprs_config.json"

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as file:
        return json.load(file)


CONFIG = load_config()

PATHS = CONFIG["paths"]
FORMULA = CONFIG["formula"]
NORMALIZATION = CONFIG["normalization"]
SEVERITY = CONFIG["severity"]
THRESHOLDS = CONFIG["risk_thresholds"]
GOVERNANCE_ACTIONS = CONFIG["governance_actions"]

STAGE2_DIR = ROOT / PATHS["stage2_results"]
IMAGE_DIR = ROOT / PATHS["document_images"]
OUTPUT_DIR = ROOT / PATHS["output_results"]


def validate_configuration():
    required_paths = (
        "stage2_results",
        "document_images",
        "output_results",
    )
    for key in required_paths:
        if key not in PATHS:
            raise ValueError(f"Missing paths.{key}")

    if not SEVERITY:
        raise ValueError("The severity configuration is empty.")

    severity_values = {}
    for label, raw_value in SEVERITY.items():
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid severity for {label}: {raw_value}")
        severity_values[label] = value

    if max(severity_values.values()) <= 0:
        raise ValueError("At least one configured severity must be positive.")

    for key in ("region_weight", "distance_weight"):
        if key not in FORMULA:
            raise ValueError(f"Missing formula.{key}")
        value = float(FORMULA[key])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"formula.{key} must be between 0 and 1.")

    region_weight = float(FORMULA["region_weight"])
    distance_weight = float(FORMULA["distance_weight"])

    if not math.isclose(
        region_weight + distance_weight, 1.0, rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError(
            "region_weight + distance_weight must equal 1."
        )

    if FORMULA.get("distance_function") != "exponential":
        raise ValueError(
            "This implementation requires distance_function='exponential'."
        )

    decay = float(FORMULA["distance_decay_ratio"])
    if not math.isfinite(decay) or decay <= 0:
        raise ValueError("distance_decay_ratio must be positive.")

    kappa = float(NORMALIZATION["scale_kappa"])
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("normalization.scale_kappa must be positive.")

    for key in ("low", "medium", "high"):
        if key not in THRESHOLDS:
            raise ValueError(f"Missing risk_thresholds.{key}")
        value = float(THRESHOLDS[key])
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Threshold {key} must be between 0 and 100.")

    if not (
        float(THRESHOLDS["low"])
        < float(THRESHOLDS["medium"])
        < float(THRESHOLDS["high"])
    ):
        raise ValueError("Thresholds must satisfy low < medium < high.")

    for category in ("Low", "Medium", "High", "Critical"):
        if category not in GOVERNANCE_ACTIONS:
            raise ValueError(f"Missing governance action for {category}.")


def get_severity(pii_type):
    if pii_type not in SEVERITY:
        raise ValueError(
            f"Unknown PII type {pii_type!r}. Add an evidence-based severity "
            "mapping to config/lcprs_config.json; no value was guessed."
        )
    return float(SEVERITY[pii_type])


def get_sensitivity(pii_type):
    maximum = max(float(value) for value in SEVERITY.values())
    return get_severity(pii_type) / maximum


def get_visibility(item):
    """
    Use an explicit item visibility value if available.
    Otherwise use 1.0 as a transparent assumption, not a measured value.
    """
    raw_value = item.get("visibility")
    source = "item"

    if raw_value is None:
        raw_value = 1.0
        source = "assumed_default"

    value = float(raw_value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Visibility must be in [0, 1], got {raw_value!r}")

    return value, source


def bbox_center(bbox):
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def normalized_distance(item_a, item_b, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Page dimensions must be positive.")

    ax, ay = bbox_center(item_a["pii_bbox"])
    bx, by = bbox_center(item_b["pii_bbox"])

    page_diagonal = math.hypot(width, height)
    center_distance = math.hypot(ax - bx, ay - by)

    return center_distance / page_diagonal


def layout_linkage(item_a, item_b, width, height):
    region_a = item_a.get("layout_region_id")
    region_b = item_b.get("layout_region_id")

    same_region = (
        region_a is not None
        and region_b is not None
        and region_a == region_b
    )
    region_cue = 1.0 if same_region else 0.0

    distance = normalized_distance(item_a, item_b, width, height)
    decay = float(FORMULA["distance_decay_ratio"])
    proximity_cue = math.exp(-distance / decay)

    region_weight = float(FORMULA["region_weight"])
    distance_weight = float(FORMULA["distance_weight"])

    linkage = (
        region_weight * region_cue
        + distance_weight * proximity_cue
    )

    return {
        "same_region": same_region,
        "normalized_distance": distance,
        "region_cue": region_cue,
        "proximity_cue": proximity_cue,
        "linkage_score": max(0.0, min(1.0, linkage)),
    }


def determine_risk_category(score):
    if score <= float(THRESHOLDS["low"]):
        return "Low"
    if score <= float(THRESHOLDS["medium"]):
        return "Medium"
    if score <= float(THRESHOLDS["high"]):
        return "High"
    return "Critical"


def calculate_lcprs(items, width, height):
    if not items:
        return {
            "score": None,
            "risk_category": "No PII detected",
            "governance_action": (
                "Review detection coverage. No detections do not prove "
                "that the document contains no sensitive information."
            ),
            "item_risk": 0.0,
            "pairwise_risk": 0.0,
            "raw_risk": 0.0,
            "item_details": [],
            "pair_details": [],
            "visibility_assumption_count": 0,
        }

    item_details = []
    prepared = []

    for item in items:
        pii_type = item["pii_type"]
        sensitivity = get_sensitivity(pii_type)
        visibility, visibility_source = get_visibility(item)
        exposure = sensitivity * visibility

        prepared.append({
            "item": item,
            "sensitivity": sensitivity,
            "visibility": visibility,
            "exposure": exposure,
        })

        item_details.append({
            "pii_type": pii_type,
            "pii_text": item.get("pii_text", ""),
            "severity": get_severity(pii_type),
            "normalized_sensitivity": sensitivity,
            "visibility": visibility,
            "visibility_source": visibility_source,
            "item_risk_contribution": exposure,
            "layout_region_id": item.get("layout_region_id"),
        })

    # R_item = sum(beta_i * V_i)
    item_risk = sum(entry["exposure"] for entry in prepared)

    # R_pair = sum over unique pairs of (beta_i*V_i + beta_j*V_j) * L_ij
    pairwise_risk = 0.0
    pair_details = []

    for left, right in combinations(prepared, 2):
        item_a = left["item"]
        item_b = right["item"]

        linkage = layout_linkage(item_a, item_b, width, height)
        pair_contribution = (
            left["exposure"] + right["exposure"]
        ) * linkage["linkage_score"]

        pairwise_risk += pair_contribution

        pair_details.append({
            "pii_a": item_a.get("pii_text", ""),
            "pii_type_a": item_a["pii_type"],
            "pii_b": item_b.get("pii_text", ""),
            "pii_type_b": item_b["pii_type"],
            **linkage,
            "pair_risk_contribution": pair_contribution,
        })

    raw_risk = item_risk + pairwise_risk
    kappa = float(NORMALIZATION["scale_kappa"])

    # LCPRS = 100 * (1 - exp(-R_raw / kappa))
    score = 100.0 * (1.0 - math.exp(-raw_risk / kappa))
    score = max(0.0, min(100.0, score))

    category = determine_risk_category(score)

    return {
        "score": score,
        "risk_category": category,
        "governance_action": GOVERNANCE_ACTIONS[category],
        "item_risk": item_risk,
        "pairwise_risk": pairwise_risk,
        "raw_risk": raw_risk,
        "item_details": item_details,
        "pair_details": pair_details,
        "visibility_assumption_count": sum(
            detail["visibility_source"] == "assumed_default"
            for detail in item_details
        ),
    }


def get_image_dimensions(document_name, stage2_data):
    width = stage2_data.get("image_width")
    height = stage2_data.get("image_height")

    if width is not None and height is not None:
        width, height = float(width), float(height)
        if (
            math.isfinite(width)
            and math.isfinite(height)
            and width > 0
            and height > 0
        ):
            return width, height

    image_path = IMAGE_DIR / f"{document_name}.png"
    if image_path.exists():
        from PIL import Image

        with Image.open(image_path) as image:
            return float(image.width), float(image.height)

    raise ValueError(
        f"Missing valid image dimensions for {document_name}; "
        "no corresponding image was found."
    )


def process_document(stage2_file):
    document_name = stage2_file.stem.removesuffix("_layout")

    with open(stage2_file, "r", encoding="utf-8-sig") as file:
        stage2_data = json.load(file)

    mappings = stage2_data.get("pii_mappings")
    if mappings is None:
        raise ValueError(
            f"{stage2_file.name} does not contain 'pii_mappings'."
        )

    valid_items = []
    skipped_missing_bbox = 0

    for index, item in enumerate(mappings):
        if not isinstance(item, dict):
            raise ValueError(f"Mapping {index} is not a JSON object.")

        bbox = item.get("pii_bbox")
        if bbox is None:
            skipped_missing_bbox += 1
            continue

        if not item.get("pii_type"):
            raise ValueError(f"Mapping {index} has no pii_type.")

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox in mapping {index}: {bbox!r}")

        coordinates = [float(value) for value in bbox]
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"Non-finite bbox in mapping {index}.")

        x1, y1, x2, y2 = coordinates
        if x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid bbox coordinate order: {bbox!r}")

        # Store validated numeric coordinates for consistent calculations.
        item = dict(item)
        item["pii_bbox"] = coordinates

        # Fail instead of inventing a severity for unknown PII labels.
        get_severity(item["pii_type"])
        valid_items.append(item)

    width, height = get_image_dimensions(document_name, stage2_data)
    result = calculate_lcprs(valid_items, width, height)

    output = {
        "lcprs_version": "3.0-proposed-layout-pairwise",
        "document": document_name,
        "image_width": width,
        "image_height": height,
        "formula": {
            "item_risk": "sum(beta_i * V_i)",
            "pairwise_risk": (
                "sum_{i<j}((beta_i*V_i + beta_j*V_j) * L_ij)"
            ),
            "linkage": (
                "region_weight*c_region + distance_weight*c_prox"
            ),
            "distance": "exp(-normalized_distance/distance_decay_ratio)",
            "final_score": "100*(1-exp(-raw_risk/scale_kappa))",
        },
        "research_status": (
            "Sensitivity-visibility is motivated by published privacy-scoring "
            "work. Pairwise layout linkage and final score scaling are proposed "
            "model components requiring empirical validation."
        ),
        "parameters": {
            "region_weight": float(FORMULA["region_weight"]),
            "distance_weight": float(FORMULA["distance_weight"]),
            "distance_function": FORMULA["distance_function"],
            "distance_decay_ratio": float(FORMULA["distance_decay_ratio"]),
            "scale_kappa": float(NORMALIZATION["scale_kappa"]),
        },
        "number_of_pii": len(valid_items),
        "number_skipped_missing_bbox": skipped_missing_bbox,
        "layout_mapping_coverage": stage2_data.get(
            "layout_mapping_coverage"
        ),
        "normalized_lcprs_score": result["score"],
        "risk_category": result["risk_category"],
        "governance_action": result["governance_action"],
        "item_risk": result["item_risk"],
        "pairwise_risk": result["pairwise_risk"],
        "raw_risk": result["raw_risk"],
        "visibility_assumption_count": result[
            "visibility_assumption_count"
        ],
        "item_details": result["item_details"],
        "pair_details": result["pair_details"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{document_name}_lcprs.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False, allow_nan=False)

    score_text = (
        "No PII scored"
        if result["score"] is None
        else f"{result['score']:.2f}/100"
    )

    print(
        f"{document_name}: PII={len(valid_items)}, "
        f"score={score_text}, category={result['risk_category']}"
    )
    print(f"Saved: {output_path}")

    if result["visibility_assumption_count"]:
        print(
            "WARNING: Visibility defaulted to 1.0 for "
            f"{result['visibility_assumption_count']} entities. "
            "This is an assumption, not a measured visibility value."
        )

    return output


def main():
    validate_configuration()

    if not STAGE2_DIR.exists():
        raise FileNotFoundError(f"Stage 2 folder not found: {STAGE2_DIR}")

    files = sorted(STAGE2_DIR.glob("*_layout.json"))
    if not files:
        raise FileNotFoundError(
            f"No *_layout.json files found in {STAGE2_DIR}"
        )

    print("LCPRS v3 — proposed layout-aware pairwise model")
    print(f"Input : {STAGE2_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    errors = []
    succeeded = 0

    for path in files:
        try:
            process_document(path)
            succeeded += 1
        except Exception as exc:
            print(f"ERROR {path.name}: {exc}")
            errors.append((path.name, str(exc)))

    print(f"\nCompleted: {succeeded}/{len(files)} documents.")
    print(f"Results: {OUTPUT_DIR}")

    if errors:
        print(f"Failed documents: {len(errors)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
