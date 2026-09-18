import json
import math
from pathlib import Path
from itertools import combinations


# ============================================================
# PATHS
# ============================================================

CONFIG_PATH = Path("../config/lcprs_config.json")
STAGE2_DIR = Path("../evaluation/stage2_results")
OUTPUT_DIR = Path("../evaluation/lcprs_results")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

print("=" * 70)
print("LCPRS - LAYOUT-CONTEXTUAL PRIVACY RISK SCORING")
print("=" * 70)

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:
    config = json.load(f)


WEIGHTS = config["weights"]
SEVERITY = config["severity"]
THRESHOLDS = config["risk_thresholds"]


print("\nLCPRS configuration loaded.")


# ============================================================
# SEVERITY
# ============================================================

def get_severity(pii_type):
    """
    Return severity assigned to a PII type.
    """

    return float(
        SEVERITY.get(
            pii_type,
            1.0
        )
    )


# ============================================================
# BOX CENTER
# ============================================================

def box_center(box):
    """
    Calculate center of bounding box.

    box:
    [x1, y1, x2, y2]
    """

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


# ============================================================
# PAGE DISTANCE
# ============================================================

def calculate_distance(box_a, box_b):
    """
    Euclidean distance between the centers
    of two PII bounding boxes.
    """

    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)

    return math.sqrt(
        (ax - bx) ** 2 +
        (ay - by) ** 2
    )


# ============================================================
# NORMALIZED PAGE POSITION CLOSENESS
# ============================================================

def calculate_position_closeness(
    box_a,
    box_b,
    image_width,
    image_height
):
    """
    Converts physical page distance into a
    normalized closeness value between 0 and 1.

    1 = very close
    0 = very far
    """

    distance = calculate_distance(
        box_a,
        box_b
    )

    page_diagonal = math.sqrt(
        image_width ** 2 +
        image_height ** 2
    )

    if page_diagonal == 0:
        return 0.0

    normalized_distance = (
        distance /
        page_diagonal
    )

    closeness = 1.0 - normalized_distance

    return max(
        0.0,
        min(
            1.0,
            closeness
        )
    )


# ============================================================
# REGION RELATIONSHIP
# ============================================================

def calculate_relationship(
    pii_a,
    pii_b,
    image_width,
    image_height
):
    """
    Implements:

    Relationship(A,B)
    =
    W_region * same_region
    +
    W_dist * page_position_closeness
    """

    region_a = pii_a.get(
        "layout_region_id"
    )

    region_b = pii_b.get(
        "layout_region_id"
    )

    same_region = (
        region_a is not None
        and
        region_b is not None
        and
        region_a == region_b
    )

    same_region_value = (
        1.0
        if same_region
        else 0.0
    )

    closeness = calculate_position_closeness(
        pii_a["pii_bbox"],
        pii_b["pii_bbox"],
        image_width,
        image_height
    )

    relationship = (
        WEIGHTS["region"] *
        same_region_value
    ) + (
        WEIGHTS["distance"] *
        closeness
    )

    return {
        "same_region": same_region,
        "same_region_value": same_region_value,
        "distance_closeness": closeness,
        "relationship": relationship
    }


# ============================================================
# PAIR RISK
# ============================================================

def calculate_pair_risk(
    pii_a,
    pii_b,
    image_width,
    image_height
):
    """
    Implements:

    PairRisk(A,B)
    =
    (SeverityA + SeverityB)
    *
    Relationship(A,B)
    """

    severity_a = get_severity(
        pii_a["pii_type"]
    )

    severity_b = get_severity(
        pii_b["pii_type"]
    )

    relationship_data = calculate_relationship(
        pii_a,
        pii_b,
        image_width,
        image_height
    )

    pair_risk = (
        severity_a +
        severity_b
    ) * relationship_data["relationship"]

    return {
        "pii_a": pii_a["pii_text"],
        "pii_a_type": pii_a["pii_type"],
        "pii_a_severity": severity_a,

        "pii_b": pii_b["pii_text"],
        "pii_b_type": pii_b["pii_type"],
        "pii_b_severity": severity_b,

        "same_region":
            relationship_data["same_region"],

        "distance_closeness":
            relationship_data["distance_closeness"],

        "relationship":
            relationship_data["relationship"],

        "pair_risk":
            pair_risk
    }


# ============================================================
# RISK CATEGORY
# ============================================================

def risk_category(normalized_score):
    """
    Convert normalized risk score into
    Low / Medium / High / Critical.
    """

    if normalized_score < THRESHOLDS["low"]:
        return "Low"

    elif normalized_score < THRESHOLDS["medium"]:
        return "Medium"

    elif normalized_score < THRESHOLDS["high"]:
        return "High"

    else:
        return "Critical"


# ============================================================
# GOVERNANCE ACTION
# ============================================================

def governance_action(category):

    actions = {
        "Low":
            "Normal handling; standard document controls.",

        "Medium":
            "Apply controlled access and review before sharing.",

        "High":
            "Restrict access and review before external sharing.",

        "Critical":
            "Restrict access, protect sensitive content, and require authorization before sharing."
    }

    return actions[category]


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(stage2_file):

    print("\n" + "=" * 70)
    print(
        f"PROCESSING {stage2_file.name}"
    )
    print("=" * 70)

    with open(
        stage2_file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    pii_items = data.get(
        "pii_mappings",
        []
    )

    # --------------------------------------------------------
    # Only use PII with a valid bounding box
    # --------------------------------------------------------

    valid_pii = [
        item
        for item in pii_items
        if item.get("pii_bbox") is not None
    ]

    print(
        f"PII items available: "
        f"{len(valid_pii)}"
    )

    # --------------------------------------------------------
    # Get image dimensions
    # --------------------------------------------------------

    image_width = 1025
    image_height = 1675

    # --------------------------------------------------------
    # Individual severities
    # --------------------------------------------------------

    standalone_risk = sum(
        get_severity(
            item["pii_type"]
        )
        for item in valid_pii
    )

    # --------------------------------------------------------
    # Pair risks
    # --------------------------------------------------------

    pair_results = []

    for pii_a, pii_b in combinations(
        valid_pii,
        2
    ):

        pair_result = calculate_pair_risk(
            pii_a,
            pii_b,
            image_width,
            image_height
        )

        pair_results.append(
            pair_result
        )

    pair_risk_total = sum(
        pair["pair_risk"]
        for pair in pair_results
    )

    # --------------------------------------------------------
    # Raw document risk
    # --------------------------------------------------------

    document_risk = (
        standalone_risk +
        pair_risk_total
    )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    max_possible_relationship = (
        WEIGHTS["region"] +
        WEIGHTS["distance"]
    )

    number_of_items = len(
        valid_pii
    )

    if number_of_items >= 2:

        maximum_pair_risk = 0.0

        for pii_a, pii_b in combinations(
            valid_pii,
            2
        ):

            maximum_pair_risk += (
                get_severity(
                    pii_a["pii_type"]
                )
                +
                get_severity(
                    pii_b["pii_type"]
                )
            ) * max_possible_relationship

    else:

        maximum_pair_risk = 0.0

    maximum_standalone_risk = sum(
        get_severity(
            item["pii_type"]
        )
        for item in valid_pii
    )

    maximum_possible_risk = (
        maximum_standalone_risk +
        maximum_pair_risk
    )

    if maximum_possible_risk > 0:

        normalized_score = (
            document_risk /
            maximum_possible_risk
        ) * 100

    else:

        normalized_score = 0.0

    normalized_score = max(
        0.0,
        min(
            100.0,
            normalized_score
        )
    )

    category = risk_category(
        normalized_score
    )

    action = governance_action(
        category
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    result = {

        "document":
            data.get(
                "document",
                stage2_file.stem
            ),

        "number_of_pii":
            len(valid_pii),

        "standalone_risk":
            standalone_risk,

        "pair_risk_total":
            pair_risk_total,

        "document_risk":
            document_risk,

        "normalized_lcprs_score":
            normalized_score,

        "risk_category":
            category,

        "governance_action":
            action,

        "pii_items":
            valid_pii,

        "pair_results":
            pair_results
    }

    output_file = (
        OUTPUT_DIR /
        stage2_file.name.replace(
            "_layout.json",
            "_lcprs.json"
        )
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        f"PII items used       : "
        f"{len(valid_pii)}"
    )

    print(
        f"Standalone risk      : "
        f"{standalone_risk:.4f}"
    )

    print(
        f"Pair risk total      : "
        f"{pair_risk_total:.4f}"
    )

    print(
        f"Document risk        : "
        f"{document_risk:.4f}"
    )

    print(
        f"LCPRS score          : "
        f"{normalized_score:.2f} / 100"
    )

    print(
        f"Risk category        : "
        f"{category}"
    )

    print(
        f"Governance action    : "
        f"{action}"
    )

    print(
        f"Saved                : "
        f"{output_file}"
    )


# ============================================================
# RUN ALL STAGE 2 DOCUMENTS
# ============================================================

stage2_files = sorted(
    STAGE2_DIR.glob(
        "*_layout.json"
    )
)

if not stage2_files:

    print(
        "\nERROR: No Stage 2 files found."
    )

    print(
        "Expected files inside:"
    )

    print(
        "evaluation/stage2_results/"
    )

    raise SystemExit(1)


for stage2_file in stage2_files:

    process_document(
        stage2_file
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("LCPRS PROCESSING COMPLETED")
print("=" * 70)

print(
    "\nResults saved in:"
)

print(
    "evaluation/lcprs_results/"
)