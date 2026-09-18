import os
import json
import re
from difflib import SequenceMatcher

from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from gliner2 import GLiNER2


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = "evaluation/stage1_images"
GT_DIR = "evaluation/stage1_ground_truth"
OUTPUT_DIR = "evaluation/stage1_results"

LABEL_PATH = "config/pii_labels.json"

MODEL_NAME = (
    "fastino/gliner2-privacy-filter-PII-multi"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD PII LABELS
# ============================================================

with open(
    LABEL_PATH,
    "r",
    encoding="utf-8"
) as f:

    PII_LABELS = json.load(f)

print(
    f"Loaded {len(PII_LABELS)} PII labels."
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    return re.sub(
        r"[^a-zA-Z0-9]",
        "",
        str(text)
    ).lower()


# ============================================================
# TOKEN NORMALIZATION
# ============================================================

def normalized_tokens(text):

    return re.findall(
        r"[a-zA-Z0-9]+",
        str(text).lower()
    )


# ============================================================
# FLATTEN GLINER OUTPUT
# ============================================================

def flatten_entities(result):

    entities = []

    if not isinstance(
        result,
        dict
    ):
        return entities

    groups = result.get(
        "entities",
        {}
    )

    if not isinstance(
        groups,
        dict
    ):
        return entities

    for label, values in groups.items():

        if not isinstance(
            values,
            list
        ):
            continue

        for value in values:

            # Simple string
            if isinstance(
                value,
                str
            ):

                entities.append(
                    {
                        "text": value,
                        "type": label,
                        "score": 0.0
                    }
                )

            # Dictionary
            elif isinstance(
                value,
                dict
            ):

                text = (
                    value.get("text")
                    or value.get("entity")
                    or value.get("value")
                )

                if not text:
                    continue

                score = value.get(
                    "score",
                    value.get(
                        "confidence",
                        0.0
                    )
                )

                entities.append(
                    {
                        "text": str(text),
                        "type": label,
                        "score": float(score)
                    }
                )

    return entities


# ============================================================
# OCR WORD MATCHING
# ============================================================

def find_entity_words(
    entity_text,
    all_words
):

    target = normalize(
        entity_text
    )

    if not target:
        return []


    # ========================================================
    # PASS 1
    # Exact individual OCR word
    # ========================================================

    for word in all_words:

        if normalize(
            word["text"]
        ) == target:

            return [word]


    # ========================================================
    # PASS 2
    # Consecutive words on SAME OCR line
    # ========================================================

    lines = {}

    for word in all_words:

        line_index = word[
            "line_index"
        ]

        lines.setdefault(
            line_index,
            []
        ).append(word)


    for line_index, words in lines.items():

        for start in range(
            len(words)
        ):

            combined = ""

            selected = []

            for end in range(
                start,
                min(
                    len(words),
                    start + 30
                )
            ):

                current = normalize(
                    words[end]["text"]
                )

                if not current:
                    continue

                combined += current

                selected.append(
                    words[end]
                )

                if combined == target:

                    return selected

                if len(combined) > len(target):

                    break


    # ========================================================
    # PASS 3
    # Fuzzy matching on same OCR line
    #
    # Useful when OCR changes punctuation slightly.
    # ========================================================

    best_words = []
    best_score = 0.0

    for line_index, words in lines.items():

        for start in range(
            len(words)
        ):

            combined = ""
            selected = []

            for end in range(
                start,
                min(
                    len(words),
                    start + 30
                )
            ):

                current = normalize(
                    words[end]["text"]
                )

                if not current:
                    continue

                combined += current

                selected.append(
                    words[end]
                )

                if (
                    len(combined)
                    >=
                    max(
                        1,
                        int(
                            len(target) * 0.75
                        )
                    )
                ):

                    score = SequenceMatcher(
                        None,
                        target,
                        combined
                    ).ratio()

                    if score > best_score:

                        best_score = score
                        best_words = list(
                            selected
                        )

                if len(combined) > len(target) + 20:

                    break

    # Only accept a strong fuzzy match.
    if best_score >= 0.88:

        return best_words

    return []


# ============================================================
# UNION OCR WORD BOXES
# ============================================================

def union_boxes(words):

    if not words:
        return None

    x1 = min(
        word["bbox"][0]
        for word in words
    )

    y1 = min(
        word["bbox"][1]
        for word in words
    )

    x2 = max(
        word["bbox"][2]
        for word in words
    )

    y2 = max(
        word["bbox"][3]
        for word in words
    )

    return [
        float(x1),
        float(y1),
        float(x2),
        float(y2)
    ]


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    box_a,
    box_b
):

    if (
        not box_a
        or not box_b
    ):
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(
        ax1,
        bx1
    )

    intersection_y1 = max(
        ay1,
        by1
    )

    intersection_x2 = min(
        ax2,
        bx2
    )

    intersection_y2 = min(
        ay2,
        by2
    )

    if (
        intersection_x2 <= intersection_x1
        or
        intersection_y2 <= intersection_y1
    ):

        return 0.0

    intersection = (
        intersection_x2
        - intersection_x1
    ) * (
        intersection_y2
        - intersection_y1
    )

    area_a = (
        ax2 - ax1
    ) * (
        ay2 - ay1
    )

    area_b = (
        bx2 - bx1
    ) * (
        by2 - by1
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:

        return 0.0

    return (
        intersection
        / union
    )


# ============================================================
# LOAD PADDLEOCR
# ============================================================

print(
    "\nLoading PaddleOCR..."
)

ocr = PaddleOCR(

    lang="en",

    enable_mkldnn=False,

    return_word_box=True,

    # IMPORTANT:
    # Do not transform the synthetic image.
    # Keep OCR coordinates in original image space.
    use_doc_orientation_classify=False,

    use_doc_unwarping=False,

    use_textline_orientation=False
)

print(
    "PaddleOCR loaded."
)


# ============================================================
# LOAD GLINER2
# ============================================================

print(
    "\nLoading GLiNER2-PII..."
)

model = GLiNER2.from_pretrained(
    MODEL_NAME
)

print(
    "GLiNER2 loaded."
)


# ============================================================
# FIND IMAGES
# ============================================================

image_files = sorted(
    filename
    for filename in os.listdir(
        IMAGE_DIR
    )
    if filename.lower().endswith(
        ".png"
    )
)

if not image_files:

    print(
        "\nERROR: No PNG files found."
    )

    raise SystemExit(1)


# ============================================================
# GLOBAL COUNTERS
# ============================================================

total_ground_truth = 0
total_detected = 0
total_mapped = 0

all_results = []


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

for filename in image_files:

    document_id = os.path.splitext(
        filename
    )[0]

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"PROCESSING {document_id}"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # PATHS
    # ========================================================

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    gt_path = os.path.join(
        GT_DIR,
        f"{document_id}.json"
    )


    # ========================================================
    # GROUND TRUTH
    # ========================================================

    with open(
        gt_path,
        "r",
        encoding="utf-8"
    ) as f:

        ground_truth = json.load(f)

    gt_entities = ground_truth.get(
        "entities",
        []
    )

    total_ground_truth += len(
        gt_entities
    )

    print(
        f"\nGround-truth PII: "
        f"{len(gt_entities)}"
    )


    # ========================================================
    # OCR
    # ========================================================

    print(
        "\nRunning OCR..."
    )

    results = ocr.predict(
        input=image_path
    )

    result = results[0]

    rec_texts = result[
        "rec_texts"
    ]

    text_words = result[
        "text_word"
    ]

    text_word_boxes = result[
        "text_word_boxes"
    ]


    # ========================================================
    # COLLECT OCR WORDS
    # ========================================================

    all_words = []

    for line_index in range(
        len(text_words)
    ):

        words = text_words[
            line_index
        ]

        boxes = text_word_boxes[
            line_index
        ]

        for word, box in zip(
            words,
            boxes
        ):

            word_text = str(
                word
            )

            if not normalize(
                word_text
            ):
                continue

            try:
                bbox = [
                    float(v)
                    for v in box.tolist()
                ]
            except AttributeError:
                bbox = [
                    float(v)
                    for v in box
                ]

            all_words.append(
                {
                    "text": word_text,
                    "bbox": bbox,
                    "line_index": line_index
                }
            )

    print(
        f"OCR word boxes: "
        f"{len(all_words)}"
    )


    # ========================================================
    # GLINER INPUT
    # ========================================================

    combined_text = "\n".join(
        str(text)
        for text in rec_texts
    )


    # ========================================================
    # GLINER2
    # ========================================================

    print(
        "\nRunning GLiNER2..."
    )

    gliner_result = model.extract_entities(
        combined_text,
        PII_LABELS
    )

    detected_entities = flatten_entities(
        gliner_result
    )

    total_detected += len(
        detected_entities
    )

    print(
        f"GLiNER entities detected: "
        f"{len(detected_entities)}"
    )


    # ========================================================
    # MAP GLINER → OCR WORD BOX
    # ========================================================

    mapped = []

    for entity in detected_entities:

        entity_text = entity[
            "text"
        ]

        entity_type = entity[
            "type"
        ]

        entity_score = entity.get(
            "score",
            0.0
        )

        matched_words = find_entity_words(
            entity_text,
            all_words
        )

        predicted_bbox = union_boxes(
            matched_words
        )

        if predicted_bbox is None:

            continue

        mapped.append(
            {
                "text": entity_text,

                "type": entity_type,

                "score": entity_score,

                "ocr_words": [
                    word["text"]
                    for word in matched_words
                ],

                "ocr_word_boxes": [
                    word["bbox"]
                    for word in matched_words
                ],

                "predicted_bbox": predicted_bbox
            }
        )

    total_mapped += len(
        mapped
    )

    print(
        f"Successfully mapped: "
        f"{len(mapped)}"
    )


    # ========================================================
    # COMPARE WITH GROUND TRUTH
    # ========================================================

    comparisons = []

    used_predictions = set()

    for gt in gt_entities:

        gt_text = gt[
            "text"
        ]

        gt_type = gt[
            "type"
        ]

        gt_bbox = gt[
            "bbox"
        ]

        best_prediction = None
        best_index = None


        # ----------------------------------------------------
        # Prefer exact normalized text + type
        # ----------------------------------------------------

        for index, prediction in enumerate(
            mapped
        ):

            if index in used_predictions:
                continue

            same_text = (
                normalize(gt_text)
                ==
                normalize(
                    prediction["text"]
                )
            )

            same_type = (
                gt_type
                ==
                prediction["type"]
            )

            if (
                same_text
                and
                same_type
            ):

                best_prediction = prediction
                best_index = index
                break


        # ----------------------------------------------------
        # Fallback: exact normalized text
        # ----------------------------------------------------

        if best_prediction is None:

            for index, prediction in enumerate(
                mapped
            ):

                if index in used_predictions:
                    continue

                if (
                    normalize(gt_text)
                    ==
                    normalize(
                        prediction["text"]
                    )
                ):

                    best_prediction = prediction
                    best_index = index
                    break


        # ----------------------------------------------------
        # Prediction found
        # ----------------------------------------------------

        if best_prediction is not None:

            used_predictions.add(
                best_index
            )

            predicted_bbox = (
                best_prediction[
                    "predicted_bbox"
                ]
            )

            iou = calculate_iou(
                gt_bbox,
                predicted_bbox
            )

            comparisons.append(
                {
                    "ground_truth_text":
                        gt_text,

                    "ground_truth_type":
                        gt_type,

                    "ground_truth_bbox":
                        gt_bbox,

                    "predicted_bbox":
                        predicted_bbox,

                    "iou":
                        iou,

                    "matched_words":
                        best_prediction[
                            "ocr_words"
                        ]
                }
            )

        else:

            comparisons.append(
                {
                    "ground_truth_text":
                        gt_text,

                    "ground_truth_type":
                        gt_type,

                    "ground_truth_bbox":
                        gt_bbox,

                    "predicted_bbox":
                        None,

                    "iou":
                        0.0,

                    "matched_words":
                        []
                }
            )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(
        image
    )

    try:

        font = ImageFont.truetype(
            "C:/Windows/Fonts/arial.ttf",
            18
        )

    except Exception:

        font = ImageFont.load_default()


    for comparison in comparisons:

        gt_box = comparison[
            "ground_truth_bbox"
        ]

        predicted_box = comparison[
            "predicted_bbox"
        ]


        # ----------------------------------------------------
        # GREEN = GROUND TRUTH
        # ----------------------------------------------------

        gx1, gy1, gx2, gy2 = [
            int(round(v))
            for v in gt_box
        ]

        draw.rectangle(
            [
                gx1,
                gy1,
                gx2,
                gy2
            ],
            outline="green",
            width=2
        )


        # ----------------------------------------------------
        # RED = PREDICTED OCR BOX
        # ----------------------------------------------------

        if predicted_box:

            px1, py1, px2, py2 = [
                int(round(v))
                for v in predicted_box
            ]

            draw.rectangle(
                [
                    px1,
                    py1,
                    px2,
                    py2
                ],
                outline="red",
                width=3
            )

            label = (
                comparison[
                    "ground_truth_text"
                ]
            )

            text_bbox = draw.textbbox(
                (0, 0),
                label,
                font=font
            )

            label_width = (
                text_bbox[2]
                - text_bbox[0]
            )

            label_height = (
                text_bbox[3]
                - text_bbox[1]
            )

            label_x = px1

            label_y = py1 - (
                label_height + 4
            )

            if label_y < 0:

                label_y = py2 + 4

            if (
                label_x
                + label_width
                > image.width
            ):

                label_x = (
                    image.width
                    - label_width
                    - 5
                )

            draw.rectangle(
                [
                    label_x - 2,
                    label_y - 2,
                    label_x
                    + label_width
                    + 2,
                    label_y
                    + label_height
                    + 2
                ],
                fill="white"
            )

            draw.text(
                (
                    label_x,
                    label_y
                ),
                label,
                fill="red",
                font=font
            )


    # ========================================================
    # SAVE VISUALIZATION
    # ========================================================

    visualization_path = os.path.join(
        OUTPUT_DIR,
        f"{document_id}_stage1.png"
    )

    image.save(
        visualization_path
    )


    # ========================================================
    # SAVE JSON
    # ========================================================

    result_path = os.path.join(
        OUTPUT_DIR,
        f"{document_id}_stage1.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            comparisons,
            f,
            indent=4
        )

    all_results.extend(
        comparisons
    )

    print(
        f"Visualization saved: "
        f"{visualization_path}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "STAGE 1 RESULTS"
)

print(
    "=" * 70
)

print(
    f"Ground-truth PII entities : "
    f"{total_ground_truth}"
)

print(
    f"GLiNER detected entities  : "
    f"{total_detected}"
)

print(
    f"Successfully mapped       : "
    f"{total_mapped}"
)


# ============================================================
# MATCHED GROUND TRUTH
# ============================================================

matched_results = [
    item
    for item in all_results
    if item["predicted_bbox"] is not None
]

ious = [
    item["iou"]
    for item in matched_results
]

if ious:

    average_iou = (
        sum(ious)
        / len(ious)
    )

else:

    average_iou = 0.0


print(
    f"Ground-truth PII matched : "
    f"{len(matched_results)}"
)

print(
    f"Average bbox IoU         : "
    f"{average_iou:.4f}"
)


# ============================================================
# INDIVIDUAL RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "GROUND-TRUTH PII MAPPING"
)

print(
    "=" * 70
)

for item in all_results:

    print(
        "\nPII:"
    )

    print(
        f'  {item["ground_truth_text"]}'
    )

    print(
        "Type:"
    )

    print(
        f'  {item["ground_truth_type"]}'
    )

    print(
        "Ground-truth bbox:"
    )

    print(
        f'  {item["ground_truth_bbox"]}'
    )

    print(
        "Predicted bbox:"
    )

    print(
        f'  {item["predicted_bbox"]}'
    )

    print(
        "Matched OCR words:"
    )

    print(
        f'  {item["matched_words"]}'
    )

    print(
        "IoU:"
    )

    print(
        f'  {item["iou"]:.4f}'
    )


# ============================================================
# STAGE 1 DECISION
# ============================================================

print(
    "\n"
    + "=" * 70
)

mapping_rate = 0.0

if total_ground_truth > 0:

    mapping_rate = (
        len(matched_results)
        / total_ground_truth
    )


if (
    total_ground_truth > 0
    and mapping_rate >= 0.80
    and average_iou >= 0.50
):

    print(
        "STAGE 1 STATUS: PASSED"
    )

    print()

    print(
        "GLiNER2 PII detections can be "
        "mapped to PaddleOCR word-level "
        "bounding boxes."
    )

    print()

    print(
        "Stage 1 is ready for the next "
        "evaluation stage."
    )

else:

    print(
        "STAGE 1 STATUS: NOT PASSED YET"
    )

    print()

    print(
        f"Ground-truth mapping rate: "
        f"{mapping_rate:.2%}"
    )

    print(
        f"Average IoU: "
        f"{average_iou:.4f}"
    )

    print()

    print(
        "Stage 1 requires further checking."
    )

print(
    "=" * 70
)