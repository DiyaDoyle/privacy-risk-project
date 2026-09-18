from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from faker import Faker
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
DATASET_ID = "docling-project/DocLayNet-v1.1"
SPLIT = "train"

# Start small:
# 5 base pages × 3 conditions = 15 synthetic documents.
BASE_PAGES = 5

# Maximum number of DocLayNet rows to inspect.
MAX_SCAN_ROWS = 5000

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "evaluation" / "dataset"

IMG_DIR = OUT_ROOT / "images"
ANN_DIR = OUT_ROOT / "annotations"
PREVIEW_DIR = OUT_ROOT / "previews"


# ============================================================
# DOCLAYNET CATEGORY MAPPING
# ============================================================

CATEGORY_NAMES = {
    1: "caption",
    2: "footnote",
    3: "formula",
    4: "list-item",
    5: "page-footer",
    6: "page-header",
    7: "picture",
    8: "section-header",
    9: "table",
    10: "text",
    11: "title",
}

# Only inject synthetic PII into existing meaningful regions.
ELIGIBLE_CATEGORIES = {
    "text",
    "table",
    "list-item",
}


# ============================================================
# PII TYPES
# ============================================================

PII_ORDER = [
    "person",
    "email",
    "phone_number",
    "date_of_birth",
    "address",
    "passport_number",
]


# ============================================================
# RANDOMNESS
# ============================================================

random.seed(SEED)

fake = Faker("en_US")
Faker.seed(SEED)


# ============================================================
# FONT
# ============================================================

FONT_PATHS = [
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\calibri.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
]


def get_font(size: int):
    """Load a readable Windows font, with Pillow fallback."""
    for font_path in FONT_PATHS:
        if font_path.exists():
            return ImageFont.truetype(
                str(font_path),
                size=size,
            )

    return ImageFont.load_default()


# ============================================================
# SYNTHETIC PII
# ============================================================

def generate_pii_values() -> dict[str, str]:
    """
    Generate completely synthetic PII.

    No real personal information is used.
    """
    return {
        "person": fake.name(),
        "email": fake.email(),
        "phone_number": fake.phone_number(),
        "date_of_birth": fake.date_of_birth(
            minimum_age=18,
            maximum_age=75,
        ).strftime("%d %B %Y"),
        "address": fake.street_address(),
        "passport_number": fake.bothify(
            text="??######",
            letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ),
    }


# ============================================================
# BOUNDING BOX HELPERS
# ============================================================

def clamp_bbox(
    box: list[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """
    Convert DocLayNet COCO-style
    [x, y, width, height]
    into integer [x1, y1, x2, y2].
    """
    x, y, w, h = box

    x1 = max(0, min(width - 1, int(round(x))))
    y1 = max(0, min(height - 1, int(round(y))))

    x2 = max(
        x1 + 1,
        min(width, int(round(x + w))),
    )

    y2 = max(
        y1 + 1,
        min(height, int(round(y + h))),
    )

    return x1, y1, x2, y2


def region_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Calculate IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (
        (ix2 - ix1)
        * (iy2 - iy1)
    )

    area_a = (
        (ax2 - ax1)
        * (ay2 - ay1)
    )

    area_b = (
        (bx2 - bx1)
        * (by2 - by1)
    )

    union = area_a + area_b - intersection

    return 0.0 if union <= 0 else intersection / union


# ============================================================
# TEXT MEASUREMENT
# ============================================================

def measure_text(
    text: str,
    font,
) -> tuple[int, int]:
    """Return rendered text width and height."""
    dummy = Image.new(
        "RGB",
        (10, 10),
        "white",
    )

    draw = ImageDraw.Draw(dummy)

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font,
    )

    return (
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    )


def fit_text(
    text: str,
    max_width: int,
    max_height: int,
    minimum_size: int = 12,
    maximum_size: int = 17,
):
    """
    Select the largest readable font that fits.

    The complete synthetic value is preserved. We do not
    truncate the ground-truth PII text.
    """
    max_width = max(10, int(max_width))
    max_height = max(10, int(max_height))

    for size in range(
        maximum_size,
        minimum_size - 1,
        -1,
    ):
        font = get_font(size)

        width, height = measure_text(
            text,
            font,
        )

        if (
            width <= max_width
            and height <= max_height
        ):
            return font, width, height

    # If necessary, use the minimum readable font.
    font = get_font(minimum_size)

    width, height = measure_text(
        text,
        font,
    )

    return font, width, height


# ============================================================
# EXTRACT DOCLAYNET REGIONS
# ============================================================

def extract_regions(
    example: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert DocLayNet annotations into usable layout regions.

    Only text, table and list-item regions are considered.
    """
    boxes = example["bboxes"]
    category_ids = example["category_id"]

    image = example["image"]

    width, height = image.size

    regions = []

    for index, (
        box,
        category_id,
    ) in enumerate(
        zip(
            boxes,
            category_ids,
        )
    ):
        category_id = int(category_id)

        category = CATEGORY_NAMES.get(
            category_id,
            f"class_{category_id}",
        )

        if category not in ELIGIBLE_CATEGORIES:
            continue

        x1, y1, x2, y2 = clamp_bbox(
            list(box),
            width,
            height,
        )

        region_width = x2 - x1
        region_height = y2 - y1

        # Ignore tiny regions.
        if region_width < 180:
            continue

        if region_height < 90:
            continue

        regions.append(
            {
                "region_index": index,
                "category_id": category_id,
                "category": category,
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                "area": (
                    region_width
                    * region_height
                ),
                "center": [
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                ],
            }
        )

    return regions


# ============================================================
# CLUSTERED CONDITION
# ============================================================

def choose_cluster_region(
    regions: list[dict[str, Any]],
):
    """
    Select one large region that can contain six compact
    synthetic fields.
    """
    candidates = []

    for region in regions:

        x1, y1, x2, y2 = region["bbox"]

        width = x2 - x1
        height = y2 - y1

        if width >= 420 and height >= 220:
            candidates.append(region)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda region: region["area"],
    )


# ============================================================
# TWO-COLUMN CONDITION
# ============================================================

def choose_two_columns(
    regions: list[dict[str, Any]],
    page_width: int,
):
    """
    Select at least two suitable regions on the left
    and two on the right.
    """
    left_regions = []
    right_regions = []

    for region in regions:

        center_x = region["center"][0]

        if center_x < page_width * 0.45:
            left_regions.append(region)

        elif center_x > page_width * 0.55:
            right_regions.append(region)

    left_regions.sort(
        key=lambda region: region["area"],
        reverse=True,
    )

    right_regions.sort(
        key=lambda region: region["area"],
        reverse=True,
    )

    if len(left_regions) < 2:
        return None

    if len(right_regions) < 2:
        return None

    return [
        left_regions[0],
        left_regions[1],
        right_regions[0],
        right_regions[1],
    ]


# ============================================================
# SCATTERED CONDITION
# ============================================================

def choose_scattered(
    regions: list[dict[str, Any]],
    page_width: int,
    page_height: int,
):
    """
    Select six spatially separated layout regions.
    """
    candidates = sorted(
        regions,
        key=lambda region: region["area"],
        reverse=True,
    )

    if len(candidates) < 6:
        return None

    selected = [
        candidates[0]
    ]

    while len(selected) < 6:

        best_candidate = None
        best_distance = -1.0

        for candidate in candidates:

            if candidate in selected:
                continue

            overlaps = False

            for existing in selected:

                overlap = region_iou(
                    tuple(candidate["bbox"]),
                    tuple(existing["bbox"]),
                )

                if overlap >= 0.15:
                    overlaps = True
                    break

            if overlaps:
                continue

            cx, cy = candidate["center"]

            minimum_distance = min(
                math.hypot(
                    (
                        cx
                        - existing["center"][0]
                    ) / page_width,
                    (
                        cy
                        - existing["center"][1]
                    ) / page_height,
                )
                for existing in selected
            )

            if minimum_distance > best_distance:

                best_distance = minimum_distance
                best_candidate = candidate

        if best_candidate is None:
            return None

        selected.append(
            best_candidate
        )

    return selected


# ============================================================
# FIELD LABELS
# ============================================================

def field_label(
    pii_type: str,
) -> str:
    """Return a document-style field label."""
    labels = {
        "person": "Name:",
        "email": "Email:",
        "phone_number": "Phone:",
        "date_of_birth": "Date of Birth:",
        "address": "Address:",
        "passport_number": "Passport:",
    }

    return labels[pii_type]


# ============================================================
# CREATE A COMPACT PII FIELD
# ============================================================

def make_text_box(
    region: dict[str, Any],
    text: str,
    pii_type: str,
    slot: int,
    slots_in_region: int,
    condition: str,
    image_size: tuple[int, int],
):
    """
    Create a compact synthetic document field inside an
    existing DocLayNet region.

    IMPORTANT:
    The returned bbox covers ONLY the sensitive value.
    The label is not part of the PII bbox.
    """
    rx1, ry1, rx2, ry2 = region["bbox"]

    image_width, image_height = image_size

    margin = 8
    gap = 4

    region_width = rx2 - rx1
    region_height = ry2 - ry1

    available_width = max(
        40,
        region_width - (2 * margin),
    )

    available_height = max(
        30,
        region_height - (2 * margin),
    )

    label = field_label(pii_type)

    label_font = get_font(12)

    label_width, label_height = measure_text(
        label,
        label_font,
    )

    # The value remains a small portion of the region.
    max_value_width = max(
        50,
        int(available_width * 0.78),
    )

    if slots_in_region > 1:

        cell_height = (
            available_height
            / slots_in_region
        )

        max_value_height = max(
            20,
            int(cell_height - 6),
        )

    else:

        max_value_height = max(
            20,
            int(available_height * 0.30),
        )

    font, text_width, text_height = fit_text(
        text,
        max_value_width,
        max_value_height,
    )

    value_width = text_width + 6
    value_height = text_height + 6

    # --------------------------------------------------------
    # Clustered: compact 2-column × 3-row field arrangement.
    # --------------------------------------------------------

    if condition == "clustered":

        cols = 2
        rows = 3

        cell_width = available_width / cols
        cell_height = available_height / rows

        cell_x = (
            rx1
            + margin
            + (slot % cols) * cell_width
        )

        cell_y = (
            ry1
            + margin
            + (slot // cols) * cell_height
        )

        # Prefer "Label: value" on one line.
        total_width = (
            label_width
            + gap
            + value_width
        )

        if total_width <= cell_width - 8:

            field_x = int(
                cell_x
                + max(
                    0,
                    (cell_width - total_width) / 2,
                )
            )

            value_x = field_x + label_width + gap

            value_y = int(
                cell_y
                + max(
                    0,
                    (cell_height - value_height) / 2,
                )
            )

            label_x = field_x

            label_y = int(
                value_y
                + max(
                    0,
                    (value_height - label_height) / 2,
                )
            )

        else:

            # Label above value.
            label_x = int(cell_x)
            label_y = int(cell_y)

            value_x = int(cell_x)

            value_y = int(
                cell_y
                + label_height
                + 2
            )

    # --------------------------------------------------------
    # Two-column / scattered.
    # --------------------------------------------------------

    else:

        # Small deterministic offset inside the selected region.
        offset_x = (slot % 3) * 8
        offset_y = (slot % 2) * 8

        field_x = (
            rx1
            + margin
            + offset_x
        )

        field_y = (
            ry1
            + margin
            + offset_y
        )

        total_width = (
            label_width
            + gap
            + value_width
        )

        if total_width <= available_width:

            label_x = int(field_x)

            value_x = int(
                field_x
                + label_width
                + gap
            )

            label_y = int(
                field_y
                + max(
                    0,
                    (value_height - label_height) / 2,
                )
            )

            value_y = int(field_y)

        else:

            label_x = int(field_x)
            label_y = int(field_y)

            value_x = int(field_x)

            value_y = int(
                field_y
                + label_height
                + 2
            )

    # --------------------------------------------------------
    # Keep value inside target region.
    # --------------------------------------------------------

    value_x = min(
        value_x,
        rx2 - margin - value_width,
    )

    value_y = min(
        value_y,
        ry2 - margin - value_height,
    )

    value_x = max(
        rx1 + margin,
        value_x,
    )

    value_y = max(
        ry1 + margin,
        value_y,
    )

    value_x2 = min(
        rx2 - margin,
        value_x + value_width,
    )

    value_y2 = min(
        ry2 - margin,
        value_y + value_height,
    )

    # Keep label inside the image.
    label_x = max(
        0,
        min(
            int(label_x),
            image_width - 2,
        ),
    )

    label_y = max(
        0,
        min(
            int(label_y),
            image_height - 2,
        ),
    )

    # Final image boundary safety.
    value_x = max(
        0,
        min(
            int(value_x),
            image_width - 2,
        ),
    )

    value_y = max(
        0,
        min(
            int(value_y),
            image_height - 2,
        ),
    )

    value_x2 = max(
        value_x + 2,
        min(
            int(value_x2),
            image_width - 1,
        ),
    )

    value_y2 = max(
        value_y + 2,
        min(
            int(value_y2),
            image_height - 1,
        ),
    )

    return {
        "bbox": [
            value_x,
            value_y,
            value_x2,
            value_y2,
        ],
        "font": font,
        "label": label,
        "label_font": label_font,
        "label_position": [
            label_x,
            label_y,
        ],
        "visible_text": text,
    }


# ============================================================
# DRAW FIELD
# ============================================================

def draw_field(
    draw,
    field: dict[str, Any],
    show_box: bool,
):
    """
    Draw a compact synthetic document field.

    Red ground-truth boxes are only drawn when show_box=True.
    """
    x1, y1, x2, y2 = field["bbox"]

    label = field["label"]
    label_font = field["label_font"]
    font = field["font"]

    label_x, label_y = field["label_position"]

    # Determine a small background containing label + value.
    label_bbox = draw.textbbox(
        (label_x, label_y),
        label,
        font=label_font,
    )

    background = [
        min(
            label_bbox[0],
            x1,
        ) - 2,

        min(
            label_bbox[1],
            y1,
        ) - 2,

        max(
            label_bbox[2],
            x2,
        ) + 2,

        max(
            label_bbox[3],
            y2,
        ) + 2,
    ]

    # Only cover the tiny synthetic field area.
    draw.rectangle(
        background,
        fill="white",
    )

    # Label.
    draw.text(
        (
            label_x,
            label_y,
        ),
        label,
        fill="black",
        font=label_font,
    )

    # Sensitive value.
    draw.text(
        (
            x1 + 3,
            y1 + 2,
        ),
        field["visible_text"],
        fill="black",
        font=font,
    )

    # Ground-truth visualization ONLY.
    if show_box:

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline="red",
            width=3,
        )


# ============================================================
# ASSIGNMENTS
# ============================================================

def make_cluster_assignments(
    region: dict[str, Any],
):
    """Put all six entities into one region."""
    return [
        (
            region,
            index,
            6,
        )
        for index in range(6)
    ]


def make_two_column_assignments(
    selected: list[dict[str, Any]],
):
    """
    Distribute six entities across left and right regions.

    selected:
        left region 1
        left region 2
        right region 1
        right region 2
    """
    return [
        (selected[0], 0, 2),  # person
        (selected[0], 1, 2),  # email
        (selected[1], 0, 1),  # phone
        (selected[2], 0, 2),  # DOB
        (selected[2], 1, 2),  # address
        (selected[3], 0, 1),  # passport
    ]


def make_scattered_assignments(
    selected: list[dict[str, Any]],
):
    """Put each entity in a separate selected region."""
    return [
        (
            selected[index],
            0,
            1,
        )
        for index in range(6)
    ]


# ============================================================
# GENERATE ONE CONDITION
# ============================================================

def generate_condition(
    example: dict[str, Any],
    base_id: str,
    condition: str,
    regions: list[dict[str, Any]],
    output_index: int,
    values: dict[str, str],
):
    """
    Generate one synthetic document condition.

    The same six PII values are used across all three
    conditions for the same base page.
    """
    image = example["image"].convert("RGB")

    page_width, page_height = image.size

    # --------------------------------------------------------
    # Select target regions.
    # --------------------------------------------------------

    if condition == "clustered":

        cluster_region = choose_cluster_region(
            regions
        )

        if cluster_region is None:
            return False

        assignments = make_cluster_assignments(
            cluster_region
        )

    elif condition == "two_column":

        selected = choose_two_columns(
            regions,
            page_width,
        )

        if selected is None:
            return False

        assignments = make_two_column_assignments(
            selected
        )

    elif condition == "scattered":

        selected = choose_scattered(
            regions,
            page_width,
            page_height,
        )

        if selected is None:
            return False

        assignments = make_scattered_assignments(
            selected
        )

    else:

        raise ValueError(
            f"Unknown condition: {condition}"
        )

    draw = ImageDraw.Draw(image)

    entities = []

    # --------------------------------------------------------
    # Create six synthetic PII fields.
    # --------------------------------------------------------

    for index, pii_type in enumerate(PII_ORDER):

        region, slot, slots_in_region = assignments[index]

        text = values[pii_type]

        field = make_text_box(
            region=region,
            text=text,
            pii_type=pii_type,
            slot=slot,
            slots_in_region=slots_in_region,
            condition=condition,
            image_size=image.size,
        )

        draw_field(
            draw=draw,
            field=field,
            show_box=False,
        )

        entities.append(
            {
                "entity_id": (
                    f"{base_id}_"
                    f"{condition}_"
                    f"{index + 1}"
                ),

                "type": pii_type,

                "text": text,

                # Exact synthetic PII bbox.
                "bbox": list(
                    field["bbox"]
                ),

                "target_region": {
                    "region_index": (
                        region["region_index"]
                    ),

                    "category_id": (
                        region["category_id"]
                    ),

                    "category": (
                        region["category"]
                    ),

                    "bbox": list(
                        region["bbox"]
                    ),
                },

                "placement": {
                    "slot": slot,
                    "slots_in_region": slots_in_region,
                },
            }
        )

    # --------------------------------------------------------
    # Output paths.
    # --------------------------------------------------------

    image_name = (
        f"{output_index:04d}_"
        f"{base_id}_"
        f"{condition}.png"
    )

    annotation_name = (
        f"{output_index:04d}_"
        f"{base_id}_"
        f"{condition}.json"
    )

    image_path = IMG_DIR / image_name

    annotation_path = ANN_DIR / annotation_name

    preview_path = PREVIEW_DIR / image_name

    # --------------------------------------------------------
    # Save actual ML input image.
    # --------------------------------------------------------

    # No red ground-truth boxes.
    image.save(
        image_path,
        format="PNG",
    )

    # --------------------------------------------------------
    # Save preview with red ground-truth boxes.
    # --------------------------------------------------------

    preview = image.copy()

    preview_draw = ImageDraw.Draw(
        preview
    )

    for entity in entities:

        x1, y1, x2, y2 = entity["bbox"]

        preview_draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline="red",
            width=3,
        )

    preview.save(
        preview_path,
        format="PNG",
    )

    # --------------------------------------------------------
    # Source metadata.
    # --------------------------------------------------------

    metadata = example.get(
        "metadata",
        {},
    )

    # --------------------------------------------------------
    # Ground-truth annotation.
    # --------------------------------------------------------

    record = {
        "dataset_version": "phase1_v3",

        "source_dataset": DATASET_ID,

        "source_split": SPLIT,

        "source_image_id": metadata.get(
            "image_id"
        ),

        "source_page_hash": metadata.get(
            "page_hash"
        ),

        "source_original_filename": metadata.get(
            "original_filename"
        ),

        "source_page_no": metadata.get(
            "page_no"
        ),

        "doc_category": metadata.get(
            "doc_category"
        ),

        "condition": condition,

        "image": image_name,

        "image_width": image.width,

        "image_height": image.height,

        "entities": entities,

        "experimental_design": {
            "same_base_page_across_conditions": True,
            "same_pii_values_across_conditions": True,
            "controlled_variable": "PII spatial placement",
            "pii_bbox_definition": (
                "Bounding box around the synthetic "
                "sensitive value rendered in the "
                "generated image."
            ),
        },

        "note": (
            "Synthetic PII fields were inserted into "
            "existing DocLayNet text/table/list-item "
            "regions. The synthetic values are generated "
            "by Faker. The same six values are used for "
            "clustered, two_column, and scattered variants "
            "of the same base page. Red rectangles are "
            "present only in preview images."
        ),
    }

    annotation_path.write_text(
        json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Create output directories.
    # --------------------------------------------------------

    IMG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)

    print(
        "PHASE 1 - DOCLAYNET + FAKER SYNTHETIC DATASET"
    )

    print("=" * 70)

    print(
        f"Dataset: {DATASET_ID}"
    )

    print(
        f"Split: {SPLIT}"
    )

    print(
        f"Base pages requested: {BASE_PAGES}"
    )

    print()

    # --------------------------------------------------------
    # Load DocLayNet in streaming mode.
    # --------------------------------------------------------

    print(
        "Loading DocLayNet in streaming mode..."
    )

    dataset = load_dataset(
        DATASET_ID,
        split=SPLIT,
        streaming=True,
    )

    dataset = dataset.shuffle(
        seed=SEED,
        buffer_size=1000,
    )

    print(
        "DocLayNet stream ready."
    )

    print()

    selected_pages = 0

    scanned_rows = 0

    generated_documents = 0

    output_index = 1

    # --------------------------------------------------------
    # Scan source pages.
    # --------------------------------------------------------

    for example in dataset:

        scanned_rows += 1

        if scanned_rows > MAX_SCAN_ROWS:
            break

        if example.get("image") is None:
            continue

        image = example["image"].convert(
            "RGB"
        )

        page_width, page_height = image.size

        regions = extract_regions(
            example
        )

        # The page must support all three conditions.
        if choose_cluster_region(
            regions
        ) is None:
            continue

        if choose_two_columns(
            regions,
            page_width,
        ) is None:
            continue

        if choose_scattered(
            regions,
            page_width,
            page_height,
        ) is None:
            continue

        metadata = example.get(
            "metadata",
            {},
        )

        base_id = str(
            metadata.get(
                "page_hash",
                metadata.get(
                    "image_id",
                    f"page_{scanned_rows}",
                ),
            )
        )

        category = metadata.get(
            "doc_category",
            "unknown",
        )

        print(
            f"[{selected_pages + 1}/{BASE_PAGES}] "
            f"Selected page: {base_id} "
            f"| category={category}"
        )

        # IMPORTANT:
        # Generate ONE PII set for this base page.
        # The same six values are reused across:
        # clustered / two_column / scattered.
        values = generate_pii_values()

        success_count = 0

        for condition in [
            "clustered",
            "two_column",
            "scattered",
        ]:

            success = generate_condition(
                example=example,
                base_id=base_id,
                condition=condition,
                regions=regions,
                output_index=output_index,
                values=values,
            )

            if success:

                success_count += 1

                generated_documents += 1

                output_index += 1

        if success_count == 3:

            selected_pages += 1

        if selected_pages >= BASE_PAGES:
            break

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    summary = {
        "dataset_version": "phase1_v3",

        "source_dataset": DATASET_ID,

        "source_split": SPLIT,

        "seed": SEED,

        "base_pages_requested": BASE_PAGES,

        "base_pages_selected": selected_pages,

        "synthetic_documents_generated": generated_documents,

        "conditions": [
            "clustered",
            "two_column",
            "scattered",
        ],

        "pii_types": PII_ORDER,

        "rows_scanned": scanned_rows,

        "image_directory": str(
            IMG_DIR
        ),

        "annotation_directory": str(
            ANN_DIR
        ),

        "preview_directory": str(
            PREVIEW_DIR
        ),

        "experimental_design": {
            "same_base_page_across_conditions": True,
            "same_pii_values_across_conditions": True,
            "controlled_variable": "PII spatial placement",
        },
    }

    summary_path = (
        OUT_ROOT
        / "dataset_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print("=" * 70)

    print(
        "PHASE 1 COMPLETE"
    )

    print("=" * 70)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()

    print(
        f"Images: {IMG_DIR}"
    )

    print(
        f"Annotations: {ANN_DIR}"
    )

    print(
        f"Previews: {PREVIEW_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
