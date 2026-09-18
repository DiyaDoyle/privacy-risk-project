import os
import json
from faker import Faker
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURATION
# ============================================================

BASE_IMAGE = "sample.png"

OUTPUT_IMAGE_DIR = "evaluation/stage1_images"
OUTPUT_GT_DIR = "evaluation/stage1_ground_truth"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_GT_DIR, exist_ok=True)

fake = Faker("en_IN")

# Fixed seed = reproducible dataset
fake.seed_instance(20260917)


# ============================================================
# FONT
# ============================================================

FONT_PATH = "C:/Windows/Fonts/arial.ttf"

if os.path.exists(FONT_PATH):
    FONT = ImageFont.truetype(FONT_PATH, 26)
    TITLE_FONT = ImageFont.truetype(FONT_PATH, 30)
else:
    FONT = ImageFont.load_default()
    TITLE_FONT = ImageFont.load_default()


# ============================================================
# GENERATE SYNTHETIC PII
# ============================================================

def generate_pii():

    return {
        "person": fake.name(),

        "email": fake.email(),

        "phone_number": fake.phone_number(),

        "date_of_birth": fake.date_of_birth(
            minimum_age=20,
            maximum_age=60
        ).strftime("%d %B %Y"),

        "address": fake.address().replace(
            "\n",
            ", "
        ),

        "passport_number": (
            fake.random_letter().upper()
            + fake.random_letter().upper()
            + str(
                fake.random_number(
                    digits=7
                )
            )
        )
    }


# ============================================================
# EXACT RENDERED TEXT BOUNDING BOX
# ============================================================

def get_rendered_text_bbox(
    image_size,
    text,
    position,
    font
):
    """
    Calculates the bounding box of the actual visible
    rendered pixels of the PII text.
    """

    x, y = position

    mask = Image.new(
        "L",
        image_size,
        0
    )

    mask_draw = ImageDraw.Draw(mask)

    mask_draw.text(
        (x, y),
        text,
        fill=255,
        font=font
    )

    bbox = mask.getbbox()

    if bbox is None:
        return [
            int(x),
            int(y),
            int(x),
            int(y)
        ]

    return [
        int(bbox[0]),
        int(bbox[1]),
        int(bbox[2]),
        int(bbox[3])
    ]


# ============================================================
# DRAW FIELD
# ============================================================

def draw_field(
    draw,
    image,
    entity_type,
    value,
    x,
    y,
    gap
):

    label = (
        entity_type
        .replace("_", " ")
        .title()
        + ":"
    )

    # Draw label
    draw.text(
        (x, y),
        label,
        fill="black",
        font=FONT
    )

    # Get label width
    label_bbox = draw.textbbox(
        (x, y),
        label,
        font=FONT
    )

    label_width = (
        label_bbox[2]
        - label_bbox[0]
    )

    # PII starts after label
    value_x = (
        x
        + label_width
        + gap
    )

    # Draw PII
    draw.text(
        (value_x, y),
        value,
        fill="black",
        font=FONT
    )

    # Get actual rendered-pixel bbox
    bbox = get_rendered_text_bbox(
        image.size,
        value,
        (value_x, y),
        FONT
    )

    return bbox


# ============================================================
# DRAW PII PANEL
# ============================================================

def draw_pii_panel(
    image,
    layout_type,
    document_id
):

    width, height = image.size

    # Extra panel below original document
    panel_height = 650

    new_image = Image.new(
        "RGB",
        (
            width,
            height + panel_height
        ),
        "white"
    )

    new_image.paste(
        image,
        (0, 0)
    )

    draw = ImageDraw.Draw(
        new_image
    )

    panel_top = height

    # --------------------------------------------------------
    # Border
    # --------------------------------------------------------

    draw.rectangle(
        [
            30,
            panel_top + 20,
            width - 30,
            height + panel_height - 20
        ],
        outline="black",
        width=3
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    draw.text(
        (
            60,
            panel_top + 45
        ),
        f"SYNTHETIC PRIVACY TEST — "
        f"{layout_type.upper()}",
        fill="black",
        font=TITLE_FONT
    )

    # --------------------------------------------------------
    # Generate PII
    # --------------------------------------------------------

    pii = generate_pii()

    ground_truth = {
        "document_id": document_id,
        "source_document": BASE_IMAGE,
        "layout_pattern": layout_type,
        "entities": []
    }

    # ========================================================
    # CLUSTERED
    # ========================================================

    if layout_type == "clustered":

        positions = {
            "person": (
                80,
                panel_top + 120
            ),

            "email": (
                80,
                panel_top + 180
            ),

            "phone_number": (
                80,
                panel_top + 240
            ),

            "date_of_birth": (
                80,
                panel_top + 300
            ),

            "address": (
                80,
                panel_top + 360
            ),

            "passport_number": (
                80,
                panel_top + 420
            )
        }

        gap = 20

    # ========================================================
    # TWO COLUMN
    # ========================================================

    elif layout_type == "two_column":

        positions = {
            "person": (
                70,
                panel_top + 130
            ),

            # Moved inward from 600
            "email": (
                500,
                panel_top + 130
            ),

            "phone_number": (
                70,
                panel_top + 270
            ),

            "date_of_birth": (
                500,
                panel_top + 270
            ),

            "address": (
                70,
                panel_top + 410
            ),

            "passport_number": (
                500,
                panel_top + 410
            )
        }

        gap = 15

    # ========================================================
    # SCATTERED
    # ========================================================

    elif layout_type == "scattered":

        positions = {
            "person": (
                70,
                panel_top + 120
            ),

            "email": (
                400,
                panel_top + 190
            ),

            "phone_number": (
                150,
                panel_top + 320
            ),

            # Moved inward from 700
            "date_of_birth": (
                650,
                panel_top + 300
            ),

            "address": (
                350,
                panel_top + 430
            ),

            "passport_number": (
                100,
                panel_top + 520
            )
        }

        gap = 15

    else:

        raise ValueError(
            f"Unknown layout type: {layout_type}"
        )

    # ========================================================
    # DRAW ALL PII
    # ========================================================

    for entity_type, value in pii.items():

        x, y = positions[
            entity_type
        ]

        bbox = draw_field(
            draw=draw,
            image=new_image,
            entity_type=entity_type,
            value=value,
            x=x,
            y=y,
            gap=gap
        )

        ground_truth[
            "entities"
        ].append(
            {
                "text": value,
                "type": entity_type,
                "bbox": bbox
            }
        )

    return (
        new_image,
        ground_truth
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print(
    "GENERATING STAGE 1 DOCULAYNET + FAKER DATASET"
)
print("=" * 70)


# ============================================================
# CHECK BASE IMAGE
# ============================================================

if not os.path.exists(BASE_IMAGE):

    raise FileNotFoundError(
        f"\nBase image not found: {BASE_IMAGE}\n"
        f"Make sure sample.png exists in the project root."
    )


# ============================================================
# LOAD BASE IMAGE
# ============================================================

base = Image.open(
    BASE_IMAGE
).convert("RGB")

print("\nBase image loaded:")
print(f"  Size: {base.size}")


# ============================================================
# THREE TEST DOCUMENTS
# ============================================================

layouts = [
    ("doc_001", "clustered"),
    ("doc_002", "two_column"),
    ("doc_003", "scattered")
]


# ============================================================
# GENERATE
# ============================================================

for document_id, layout_type in layouts:

    print(
        f"\nGenerating {document_id}..."
    )

    image, ground_truth = draw_pii_panel(
        base.copy(),
        layout_type,
        document_id
    )

    image_path = os.path.join(
        OUTPUT_IMAGE_DIR,
        f"{document_id}.png"
    )

    gt_path = os.path.join(
        OUTPUT_GT_DIR,
        f"{document_id}.json"
    )

    image.save(
        image_path,
        format="PNG"
    )

    with open(
        gt_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            ground_truth,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"  Pattern: {layout_type}"
    )

    print(
        f"  Image: {image_path}"
    )

    print(
        f"  Ground truth: {gt_path}"
    )

    print(
        f"  PII entities: "
        f"{len(ground_truth['entities'])}"
    )

    for entity in ground_truth["entities"]:

        print(
            f"    {entity['type']}: "
            f"{entity['text']}"
        )

        print(
            f"      bbox: "
            f"{entity['bbox']}"
        )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "STAGE 1 DATASET GENERATION COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nImages saved in:"
)

print(
    f"  {OUTPUT_IMAGE_DIR}"
)

print(
    "\nGround truth saved in:"
)

print(
    f"  {OUTPUT_GT_DIR}"
)

print(
    "\nNext step:"
)

print(
    "Run stage1_mapping.py"
)

print("=" * 70)