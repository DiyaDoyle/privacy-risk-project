import json
import math
from pathlib import Path
from itertools import combinations

from PIL import Image


# ============================================================
# LCPRS V2
# Layout-Contextual Privacy Risk Scoring
#
# The algorithm is parameterized entirely through
# config/lcprs_config.json.
# ============================================================


# ============================================================
# LOAD CONFIGURATION
# ============================================================

CONFIG_PATH = Path(
    "config/lcprs_config.json"
)


with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:

    CONFIG = json.load(file)


# ============================================================
# CONFIGURATION
# ============================================================

PATHS = CONFIG["paths"]

FORMULA = CONFIG["formula"]

SEVERITY = CONFIG["severity"]

THRESHOLDS = CONFIG["risk_thresholds"]

GOVERNANCE_ACTIONS = CONFIG["governance_actions"]


STAGE2_DIR = Path(
    PATHS["stage2_results"]
)

IMAGE_DIR = Path(
    PATHS["document_images"]
)

OUTPUT_DIR = Path(
    PATHS["output_results"]
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

def validate_configuration():

    region_weight = FORMULA[
        "region_weight"
    ]

    distance_weight = FORMULA[
        "distance_weight"
    ]

    if region_weight < 0:
        raise ValueError(
            "region_weight cannot be negative."
        )

    if distance_weight < 0:
        raise ValueError(
            "distance_weight cannot be negative."
        )

    if (
        region_weight +
        distance_weight
    ) <= 0:

        raise ValueError(
            "At least one LCPRS relationship weight "
            "must be greater than zero."
        )

    distance_function = FORMULA[
        "distance_function"
    ]

    supported_functions = {
        "exponential",
        "linear"
    }

    if distance_function not in supported_functions:

        raise ValueError(
            f"Unsupported distance function: "
            f"{distance_function}. "
            f"Supported values: "
            f"{sorted(supported_functions)}"
        )

    decay_ratio = FORMULA[
        "distance_decay_ratio"
    ]

    if decay_ratio <= 0:

        raise ValueError(
            "distance_decay_ratio must be greater than zero."
        )

    required_categories = {
        "low",
        "medium",
        "high"
    }

    missing_thresholds = (
        required_categories
        -
        set(THRESHOLDS.keys())
    )

    if missing_thresholds:

        raise ValueError(
            "Missing risk thresholds: "
            f"{sorted(missing_thresholds)}"
        )

    required_actions = {
        "Low",
        "Medium",
        "High",
        "Critical"
    }

    missing_actions = (
        required_actions
        -
        set(GOVERNANCE_ACTIONS.keys())
    )

    if missing_actions:

        raise ValueError(
            "Missing governance actions: "
            f"{sorted(missing_actions)}"
        )


validate_configuration()


# ============================================================
# PII SEVERITY
# ============================================================

def get_severity(pii_type):

    if pii_type not in SEVERITY:

        raise ValueError(
            f"PII type '{pii_type}' has no configured "
            f"severity value."
        )

    severity = float(
        SEVERITY[pii_type]
    )

    if severity < 0:

        raise ValueError(
            f"Negative severity configured for "
            f"PII type '{pii_type}'."
        )

    return severity


# ============================================================
# BOUNDING BOX CENTER
# ============================================================

def get_box_center(box):

    if not box or len(box) != 4:

        raise ValueError(
            f"Invalid bounding box: {box}"
        )

    x1, y1, x2, y2 = (
        float(value)
        for value in box
    )

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


# ============================================================
# PHYSICAL PAGE DISTANCE
# ============================================================

def calculate_distance(
    box_a,
    box_b
):

    ax, ay = get_box_center(
        box_a
    )

    bx, by = get_box_center(
        box_b
    )

    return math.sqrt(
        (ax - bx) ** 2
        +
        (ay - by) ** 2
    )


# ============================================================
# PAGE-POSITION CLOSENESS
# ============================================================

def calculate_position_closeness(
    box_a,
    box_b,
    image_width,
    image_height
):
    """
    Converts physical page distance into
    a normalized closeness value in [0, 1].

    1 = very close
    0 = very far

    The mathematical form is selected from configuration.
    """

    distance = calculate_distance(
        box_a,
        box_b
    )

    page_diagonal = math.sqrt(
        image_width ** 2
        +
        image_height ** 2
    )

    if page_diagonal <= 0:

        raise ValueError(
            "Image dimensions must be positive."
        )

    normalized_distance = (
        distance /
        page_diagonal
    )

    distance_function = FORMULA[
        "distance_function"
    ]

    decay_ratio = FORMULA[
        "distance_decay_ratio"
    ]

    # --------------------------------------------------------
    # Exponential distance decay
    # --------------------------------------------------------

    if distance_function == "exponential":

        closeness = math.exp(
            -normalized_distance /
            decay_ratio
        )

    # --------------------------------------------------------
    # Linear distance decay
    # --------------------------------------------------------

    elif distance_function == "linear":

        closeness = (
            1.0 -
            normalized_distance
        )

    else:

        raise ValueError(
            f"Unsupported distance function: "
            f"{distance_function}"
        )

    return max(
        0.0,
        min(
            1.0,
            closeness
        )
    )


# ============================================================
# LAYOUT RELATIONSHIP
# ============================================================

def calculate_relationship(
    pii_a,
    pii_b,
    image_width,
    image_height
):
    """
    LCPRS relationship:

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

    closeness = (
        calculate_position_closeness(
            pii_a["pii_bbox"],
            pii_b["pii_bbox"],
            image_width,
            image_height
        )
    )

    relationship = (

        FORMULA["region_weight"]
        *
        same_region_value

    ) + (

        FORMULA["distance_weight"]
        *
        closeness
    )

    return {

        "same_region":
            same_region,

        "same_region_value":
            same_region_value,

        "distance_closeness":
            closeness,

        "relationship":
            relationship
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

    relationship_data = (
        calculate_relationship(
            pii_a,
            pii_b,
            image_width,
            image_height
        )
    )

    pair_risk = (

        severity_a
        +
        severity_b

    ) * relationship_data[
        "relationship"
    ]

    return {

        "pii_a":
            pii_a["pii_text"],

        "pii_a_type":
            pii_a["pii_type"],

        "severity_a":
            severity_a,

        "pii_b":
            pii_b["pii_text"],

        "pii_b_type":
            pii_b["pii_type"],

        "severity_b":
            severity_b,

        "same_region":
            relationship_data[
                "same_region"
            ],

        "distance_closeness":
            relationship_data[
                "distance_closeness"
            ],

        "relationship":
            relationship_data[
                "relationship"
            ],

        "pair_risk":
            pair_risk
    }


# ============================================================
# MAXIMUM POSSIBLE RISK
# ============================================================

def calculate_maximum_possible_risk(
    pii_items
):
    """
    Calculates the theoretical maximum for
    this document's PII composition.

    This is used only for normalization.

    Maximum relationship occurs when:

        same_region = 1
        closeness = 1
    """

    maximum_relationship = (

        FORMULA["region_weight"]
        +
        FORMULA["distance_weight"]
    )

    standalone_maximum = sum(

        get_severity(
            item["pii_type"]
        )

        for item in pii_items
    )

    pair_maximum = 0.0

    for pii_a, pii_b in combinations(
        pii_items,
        2
    ):

        severity_sum = (

            get_severity(
                pii_a["pii_type"]
            )

            +

            get_severity(
                pii_b["pii_type"]
            )
        )

        pair_maximum += (
            severity_sum
            *
            maximum_relationship
        )

    return (
        standalone_maximum
        +
        pair_maximum
    )


# ============================================================
# RISK CATEGORY
# ============================================================

def determine_risk_category(
    normalized_score
):

    if normalized_score < THRESHOLDS[
        "low"
    ]:

        return "Low"

    if normalized_score < THRESHOLDS[
        "medium"
    ]:

        return "Medium"

    if normalized_score < THRESHOLDS[
        "high"
    ]:

        return "High"

    return "Critical"


# ============================================================
# GOVERNANCE ACTION
# ============================================================

def get_governance_action(
    category
):

    if category not in GOVERNANCE_ACTIONS:

        raise ValueError(
            f"No governance action configured "
            f"for category '{category}'."
        )

    return GOVERNANCE_ACTIONS[
        category
    ]


# ============================================================
# GET IMAGE DIMENSIONS
# ============================================================

def get_image_dimensions(
    document_name,
    stage2_data
):
    """
    Uses dimensions stored in Stage 2 if available.

    Otherwise opens the corresponding document image.

    No image dimensions are assumed.
    """

    width = stage2_data.get(
        "image_width"
    )

    height = stage2_data.get(
        "image_height"
    )

    if width is not None and height is not None:

        width = float(width)
        height = float(height)

        if width > 0 and height > 0:

            return width, height

    image_path = (
        IMAGE_DIR /
        f"{document_name}.png"
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Cannot determine dimensions for "
            f"'{document_name}'. "
            f"Expected image: {image_path}"
        )

    with Image.open(
        image_path
    ) as image:

        width, height = image.size

    return (
        float(width),
        float(height)
    )


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(
    stage2_file
):

    document_name = (
        stage2_file.stem
        .replace(
            "_layout",
            ""
        )
    )

    print("\n" + "=" * 70)

    print(
        f"PROCESSING {document_name}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load Stage 2
    # --------------------------------------------------------

    with open(
        stage2_file,
        "r",
        encoding="utf-8"
    ) as file:

        stage2_data = json.load(
            file
        )

    pii_mappings = stage2_data.get(
        "pii_mappings"
    )

    if pii_mappings is None:

        raise ValueError(
            f"{stage2_file.name} does not contain "
            f"'pii_mappings'."
        )

    # --------------------------------------------------------
    # Keep every PII item that has a valid bounding box.
    #
    # An unmatched layout region does NOT mean the PII
    # disappears from LCPRS. It can still participate in
    # physical-distance relationships.
    # --------------------------------------------------------

    valid_pii = []

    for item in pii_mappings:

        bbox = item.get(
            "pii_bbox"
        )

        pii_type = item.get(
            "pii_type"
        )

        pii_text = item.get(
            "pii_text"
        )

        if bbox is None:

            continue

        if not pii_type:

            raise ValueError(
                f"Missing PII type in "
                f"{stage2_file.name}"
            )

        if not pii_text:

            raise ValueError(
                f"Missing PII text in "
                f"{stage2_file.name}"
            )

        # Validate that severity exists.
        get_severity(
            pii_type
        )

        valid_pii.append(
            item
        )

    print(
        f"PII items available : "
        f"{len(valid_pii)}"
    )

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    image_width, image_height = (
        get_image_dimensions(
            document_name,
            stage2_data
        )
    )

    print(
        f"Image dimensions    : "
        f"{image_width:.0f} x "
        f"{image_height:.0f}"
    )

    # --------------------------------------------------------
    # Standalone severity contribution
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

        pair_result = (
            calculate_pair_risk(
                pii_a,
                pii_b,
                image_width,
                image_height
            )
        )

        pair_results.append(
            pair_result
        )

    pair_risk_total = sum(

        item["pair_risk"]

        for item in pair_results
    )

    # --------------------------------------------------------
    # Document risk
    # --------------------------------------------------------

    document_risk = (

        standalone_risk
        +
        pair_risk_total
    )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    maximum_possible_risk = (
        calculate_maximum_possible_risk(
            valid_pii
        )
    )

    if maximum_possible_risk > 0:

        normalized_score = (

            document_risk
            /
            maximum_possible_risk

        ) * 100.0

    else:

        normalized_score = 0.0

    normalized_score = max(
        0.0,
        min(
            100.0,
            normalized_score
        )
    )

    # --------------------------------------------------------
    # Risk category
    # --------------------------------------------------------

    category = (
        determine_risk_category(
            normalized_score
        )
    )

    # --------------------------------------------------------
    # Governance recommendation
    # --------------------------------------------------------

    governance = (
        get_governance_action(
            category
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    result = {

        "lcprs_version":
            "2.0",

        "document":
            document_name,

        "image_width":
            image_width,

        "image_height":
            image_height,

        "formula": {

            "region_weight":
                FORMULA[
                    "region_weight"
                ],

            "distance_weight":
                FORMULA[
                    "distance_weight"
                ],

            "distance_function":
                FORMULA[
                    "distance_function"
                ],

            "distance_decay_ratio":
                FORMULA[
                    "distance_decay_ratio"
                ]
        },

        "number_of_pii":
            len(valid_pii),

        "standalone_risk":
            standalone_risk,

        "pair_risk_total":
            pair_risk_total,

        "document_risk":
            document_risk,

        "maximum_possible_risk":
            maximum_possible_risk,

        "normalized_lcprs_score":
            normalized_score,

        "risk_category":
            category,

        "governance_action":
            governance,

        "pii_items":
            valid_pii,

        "pair_results":
            pair_results
    }

    output_file = (

        OUTPUT_DIR
        /
        f"{document_name}_lcprs.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print(
        f"Standalone risk    : "
        f"{standalone_risk:.4f}"
    )

    print(
        f"Pair risk total    : "
        f"{pair_risk_total:.4f}"
    )

    print(
        f"Document risk      : "
        f"{document_risk:.4f}"
    )

    print(
        f"Maximum possible   : "
        f"{maximum_possible_risk:.4f}"
    )

    print(
        f"LCPRS score        : "
        f"{normalized_score:.2f} / 100"
    )

    print(
        f"Risk category      : "
        f"{category}"
    )

    print(
        f"Governance action  : "
        f"{governance}"
    )

    print(
        f"Saved              : "
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "LCPRS V2 - "
        "LAYOUT-CONTEXTUAL PRIVACY RISK SCORING"
    )
    print("=" * 70)

    print(
        "\nConfiguration loaded successfully."
    )

    print(
        f"Region weight       : "
        f"{FORMULA['region_weight']}"
    )

    print(
        f"Distance weight     : "
        f"{FORMULA['distance_weight']}"
    )

    print(
        f"Distance function   : "
        f"{FORMULA['distance_function']}"
    )

    print(
        f"Distance parameter  : "
        f"{FORMULA['distance_decay_ratio']}"
    )

    stage2_files = sorted(
        STAGE2_DIR.glob(
            "*_layout.json"
        )
    )

    if not stage2_files:

        raise FileNotFoundError(
            "No Stage 2 result files found in "
            f"{STAGE2_DIR}"
        )

    errors = []

    for stage2_file in stage2_files:

        try:

            process_document(
                stage2_file
            )

        except Exception as error:

            print(
                "\nERROR processing "
                f"{stage2_file.name}: "
                f"{error}"
            )

            errors.append(
                (
                    stage2_file.name,
                    str(error)
                )
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "LCPRS V2 PROCESSING COMPLETED"
    )
    print("=" * 70)

    print(
        "\nResults saved in:"
    )

    print(
        OUTPUT_DIR
    )

    if errors:

        print(
            f"\nDocuments with errors: "
            f"{len(errors)}"
        )

        for name, error in errors:

            print(
                f"  {name}: {error}"
            )

        raise SystemExit(1)

    print(
        "\nAll documents processed successfully."
    )


if __name__ == "__main__":

    main()