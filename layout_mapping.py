import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from paddleocr import LayoutDetection


# ============================================================
# PATHS
# ============================================================

STAGE1_RESULTS_DIR = Path("evaluation/stage1_results")
STAGE1_IMAGES_DIR = Path("evaluation/stage1_images")

OUTPUT_DIR = Path("evaluation/stage2_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("STAGE 2 - PII TO LAYOUT REGION MAPPING")
print("=" * 70)

print("\nLoading PP-DocLayout-L...")

layout_model = LayoutDetection(
    model_name="PP-DocLayout-L",
    device="cpu",
    enable_mkldnn=False
)

print("PP-DocLayout-L loaded successfully.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_box_center(box):
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def point_inside_box(point, box):

    px, py = point
    x1, y1, x2, y2 = box

    return (
        x1 <= px <= x2
        and
        y1 <= py <= y2
    )


def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def find_layout_region(pii_box, layout_regions):

    center = get_box_center(pii_box)

    # --------------------------------------------------------
    # First choice: center of PII is inside layout region
    # --------------------------------------------------------

    containing = []

    for region in layout_regions:

        if point_inside_box(
            center,
            region["bbox"]
        ):
            containing.append(region)

    if containing:

        # If several regions contain the point,
        # choose the smallest one.
        containing.sort(
            key=lambda r:
            (
                r["bbox"][2] - r["bbox"][0]
            )
            *
            (
                r["bbox"][3] - r["bbox"][1]
            )
        )

        return containing[0], "center_inside"

    # --------------------------------------------------------
    # Second choice: highest IoU
    # --------------------------------------------------------

    best_region = None
    best_iou = 0.0

    for region in layout_regions:

        iou = calculate_iou(
            pii_box,
            region["bbox"]
        )

        if iou > best_iou:

            best_iou = iou
            best_region = region

    if best_region is not None and best_iou > 0:

        return best_region, "highest_iou"

    return None, "unmatched"


def get_font(size=20):

    possible_fonts = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf"
    ]

    for font_path in possible_fonts:

        if Path(font_path).exists():

            return ImageFont.truetype(
                font_path,
                size
            )

    return ImageFont.load_default()


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

stage1_files = sorted(
    STAGE1_RESULTS_DIR.glob("*_stage1.json")
)

if not stage1_files:

    print("\nERROR: No Stage 1 JSON files found.")

    raise SystemExit(1)


total_pii = 0
total_mapped = 0


for stage1_file in stage1_files:

    document_name = stage1_file.stem.replace(
        "_stage1",
        ""
    )

    image_path = (
        STAGE1_IMAGES_DIR /
        f"{document_name}.png"
    )

    print("\n" + "=" * 70)
    print(f"PROCESSING {document_name}")
    print("=" * 70)

    if not image_path.exists():

        print(
            f"WARNING: Image not found: {image_path}"
        )

        continue

    # --------------------------------------------------------
    # Load Stage 1 results
    # --------------------------------------------------------

    with open(
        stage1_file,
        "r",
        encoding="utf-8"
    ) as f:

        pii_results = json.load(f)

    # --------------------------------------------------------
    # Run PP-DocLayout
    # --------------------------------------------------------

    print("\nRunning PP-DocLayout...")

    results = layout_model.predict(
        input=str(image_path),
        batch_size=1
    )

    layout_regions = []

    for result in results:

        data = result.json

        if isinstance(data, str):

            data = json.loads(data)

        result_data = data.get(
            "res",
            data
        )

        boxes = result_data.get(
            "boxes",
            []
        )

        for box_data in boxes:

            coordinate = box_data.get(
                "coordinate"
            )

            if coordinate is None:

                coordinate = box_data.get(
                    "bbox"
                )

            if coordinate is None:
                continue

            layout_regions.append(
                {
                    "region_id":
                        len(layout_regions),

                    "label":
                        box_data.get(
                            "label",
                            "unknown"
                        ),

                    "score":
                        float(
                            box_data.get(
                                "score",
                                0.0
                            )
                        ),

                    "bbox":
                        [
                            float(coordinate[0]),
                            float(coordinate[1]),
                            float(coordinate[2]),
                            float(coordinate[3])
                        ]
                }
            )

    print(
        f"Layout regions detected: "
        f"{len(layout_regions)}"
    )

    # --------------------------------------------------------
    # Map PII to layout
    # --------------------------------------------------------

    document_results = []

    for pii in pii_results:

        pii_box = pii.get(
            "predicted_bbox"
        )

        if pii_box is None:
            continue

        total_pii += 1

        region, method = find_layout_region(
            pii_box,
            layout_regions
        )

        if region is not None:

            total_mapped += 1

            document_results.append(
                {
                    "pii_text":
                        pii.get(
                            "ground_truth_text",
                            ""
                        ),

                    "pii_type":
                        pii.get(
                            "ground_truth_type",
                            ""
                        ),

                    "pii_bbox":
                        pii_box,

                    "layout_region_id":
                        region["region_id"],

                    "layout_region_type":
                        region["label"],

                    "layout_region_score":
                        region["score"],

                    "layout_region_bbox":
                        region["bbox"],

                    "mapping_method":
                        method
                }
            )

        else:

            document_results.append(
                {
                    "pii_text":
                        pii.get(
                            "ground_truth_text",
                            ""
                        ),

                    "pii_type":
                        pii.get(
                            "ground_truth_type",
                            ""
                        ),

                    "pii_bbox":
                        pii_box,

                    "layout_region_id":
                        None,

                    "layout_region_type":
                        None,

                    "layout_region_score":
                        None,

                    "layout_region_bbox":
                        None,

                    "mapping_method":
                        "unmatched"
                }
            )

    # ========================================================
    # CREATE VISUALIZATION
    # ========================================================

    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(
        image
    )

    region_font = get_font(18)
    pii_font = get_font(20)

    # --------------------------------------------------------
    # Draw layout regions
    # --------------------------------------------------------

    for region in layout_regions:

        x1, y1, x2, y2 = [
            int(v)
            for v in region["bbox"]
        ]

        label = (
            f'{region["region_id"]}: '
            f'{region["label"]}'
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="blue",
            width=2
        )

        draw.text(
            (x1, max(0, y1 - 20)),
            label,
            fill="blue",
            font=region_font
        )

    # --------------------------------------------------------
    # Draw PII boxes
    # --------------------------------------------------------

    for item in document_results:

        box = item["pii_bbox"]

        x1, y1, x2, y2 = [
            int(v)
            for v in box
        ]

        if item["layout_region_id"] is not None:

            outline = "red"

            label = (
                f'PII: {item["pii_type"]} '
                f'→ region '
                f'{item["layout_region_id"]}'
            )

        else:

            outline = "orange"

            label = (
                f'PII: {item["pii_type"]} '
                f'→ UNMATCHED'
            )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=outline,
            width=3
        )

        draw.text(
            (x1, y2 + 2),
            label,
            fill=outline,
            font=pii_font
        )

    # --------------------------------------------------------
    # Save visualization
    # --------------------------------------------------------

    output_image = (
        OUTPUT_DIR /
        f"{document_name}_layout.png"
    )

    image.save(
        output_image
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output_json = (
        OUTPUT_DIR /
        f"{document_name}_layout.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "document":
                    document_name,

                "layout_regions":
                    layout_regions,

                "pii_mappings":
                    document_results
            },
            f,
            indent=2
        )

    mapped_count = sum(
        1
        for item in document_results
        if item["layout_region_id"] is not None
    )

    print(
        f"PII items processed: "
        f"{len(document_results)}"
    )

    print(
        f"PII mapped to layout regions: "
        f"{mapped_count}"
    )

    print(
        f"Saved JSON: {output_json}"
    )

    print(
        f"Saved image: {output_image}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 SUMMARY")
print("=" * 70)

print(
    f"Total PII boxes processed : "
    f"{total_pii}"
)

print(
    f"PII mapped to layout      : "
    f"{total_mapped}"
)

if total_pii > 0:

    mapping_rate = (
        total_mapped /
        total_pii
    ) * 100

    print(
        f"Layout mapping rate       : "
        f"{mapping_rate:.2f}%"
    )

print("\nSTAGE 2 COMPLETED")

print(
    "\nVisual and JSON results are in:"
)

print(
    "evaluation/stage2_results/"
)