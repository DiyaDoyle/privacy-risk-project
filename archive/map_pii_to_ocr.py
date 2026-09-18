import os
import json
import re

from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from gliner2 import GLiNER2


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = "../sample.png"

OUTPUT_DIR = "mapping_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_PATH = "../config/pii_labels.json"

MODEL_NAME = "fastino/gliner2-privacy-filter-PII-multi"


# ============================================================
# LOAD PII LABELS
# ============================================================

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    PII_LABELS = json.load(f)

print(f"Loaded {len(PII_LABELS)} PII labels.")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text only for locating OCR words.

    IMPORTANT:
    GLiNER still uses the ORIGINAL OCR text.
    Normalization is used only to locate the corresponding
    PaddleOCR word inside that original text.
    """

    return re.sub(
        r"[^a-zA-Z0-9]",
        "",
        str(text)
    ).lower()


# ============================================================
# FIND WORD POSITION INSIDE ORIGINAL OCR LINE
# ============================================================

def locate_word_in_line(line_text, word_text, search_start):
    """
    Locate a PaddleOCR word inside the ORIGINAL rec_text line.

    Returns:
        (start, end)

    These positions refer directly to line_text.
    """

    normalized_word = normalize_text(word_text)

    if not normalized_word:
        return None, None

    # Search from the current position.
    normalized_chars = []
    original_positions = []

    for i in range(
        search_start,
        len(line_text)
    ):

        char = line_text[i]

        if char.isalnum():

            normalized_chars.append(
                char.lower()
            )

            original_positions.append(i)

    normalized_remaining = "".join(
        normalized_chars
    )

    position = normalized_remaining.find(
        normalized_word
    )

    if position == -1:

        return None, None

    start_index = original_positions[
        position
    ]

    end_index = original_positions[
        position + len(normalized_word) - 1
    ] + 1

    return start_index, end_index


# ============================================================
# PADDLEOCR
# ============================================================

print("\nLoading PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    return_word_box=True
)

print("PaddleOCR loaded successfully.")


# ============================================================
# RUN OCR
# ============================================================

print("\nRunning OCR...")

results = ocr.predict(
    input=IMAGE_PATH
)

result = results[0]


# ============================================================
# EXTRACT OCR OUTPUT
# ============================================================

rec_texts = result["rec_texts"]
rec_boxes = result["rec_boxes"]

text_words = result["text_word"]
text_word_boxes = result["text_word_boxes"]

print(
    f"OCR regions collected: "
    f"{len(rec_texts)}"
)

print(
    f"OCR word groups collected: "
    f"{len(text_words)}"
)


# ============================================================
# BUILD OCR DATA
# ============================================================

ocr_regions = []

combined_text_parts = []

current_position = 0


for line_index, line_text in enumerate(
    rec_texts
):

    line_text = str(line_text)

    # --------------------------------------------------------
    # GLOBAL GLINER CHARACTER POSITIONS
    # --------------------------------------------------------

    line_start = current_position

    line_end = (
        line_start
        + len(line_text)
    )

    line_box = rec_boxes[
        line_index
    ].tolist()

    # --------------------------------------------------------
    # WORDS FROM PADDLEOCR
    # --------------------------------------------------------

    words = text_words[
        line_index
    ]

    word_boxes = text_word_boxes[
        line_index
    ]

    word_data = []

    search_position = 0


    for word_index, (
        word,
        box
    ) in enumerate(
        zip(words, word_boxes)
    ):

        word_text = str(word)

        # ----------------------------------------------------
        # Locate this exact OCR word inside rec_texts.
        # ----------------------------------------------------

        word_start, word_end = (
            locate_word_in_line(
                line_text,
                word_text,
                search_position
            )
        )

        # ----------------------------------------------------
        # If found, continue searching after it.
        # ----------------------------------------------------

        if word_end is not None:

            search_position = word_end


        word_data.append({

            "index": word_index,

            "text": word_text,

            "normalized": normalize_text(
                word_text
            ),

            "bbox": [
                float(v)
                for v in box.tolist()
            ],

            "start": word_start,

            "end": word_end

        })


    # --------------------------------------------------------
    # SAVE OCR REGION
    # --------------------------------------------------------

    ocr_regions.append({

        "index": line_index,

        "text": line_text,

        "start": line_start,

        "end": line_end,

        "bbox": line_box,

        "words": word_data

    })


    combined_text_parts.append(
        line_text
    )


    # +1 = newline between OCR lines
    current_position = (
        line_end + 1
    )


# ============================================================
# CREATE EXACT TEXT GIVEN TO GLINER
# ============================================================

combined_text = "\n".join(
    combined_text_parts
)


# ============================================================
# SAVE OCR TEXT
# ============================================================

ocr_text_path = os.path.join(
    OUTPUT_DIR,
    "combined_ocr_text.txt"
)

with open(
    ocr_text_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        combined_text
    )

print(
    "Combined OCR text saved."
)


# ============================================================
# GLINER2-PII
# ============================================================

print("\nLoading GLiNER2-PII...")

model = GLiNER2.from_pretrained(
    MODEL_NAME
)

print(
    "GLiNER2-PII loaded successfully."
)


# ============================================================
# PII DETECTION
# ============================================================

print(
    "\nRunning PII detection..."
)

pii_results = model.extract_entities(
    combined_text,
    PII_LABELS
)


# ============================================================
# FLATTEN GLINER OUTPUT
# ============================================================

def flatten_entities(data):

    entities = []

    if not isinstance(
        data,
        dict
    ):

        return entities


    entity_groups = data.get(
        "entities",
        {}
    )


    if not isinstance(
        entity_groups,
        dict
    ):

        return entities


    for label, entity_list in (
        entity_groups.items()
    ):

        if not isinstance(
            entity_list,
            list
        ):

            continue


        for item in entity_list:

            if not isinstance(
                item,
                dict
            ):

                continue


            item = dict(item)

            # Preserve the label
            # from config/pii_labels.json

            item["label"] = label

            entities.append(
                item
            )


    return entities


entities = flatten_entities(
    pii_results
)

print(
    f"PII entities detected: "
    f"{len(entities)}"
)


# ============================================================
# ENTITY HELPERS
# ============================================================

def get_entity_text(entity):

    return (
        entity.get("text")
        or entity.get("entity")
        or entity.get("value")
        or ""
    )


def get_entity_label(entity):

    return (
        entity.get("label")
        or entity.get("type")
        or entity.get("entity_type")
        or "unknown"
    )


def get_entity_score(entity):

    value = (
        entity.get("score")
        if entity.get("score") is not None
        else entity.get("confidence", 0)
    )

    try:

        return float(value)

    except:

        return 0.0


def get_entity_start(entity):

    value = (
        entity.get("start")
        if entity.get("start") is not None
        else entity.get("char_start")
    )

    return int(value)


def get_entity_end(entity):

    value = (
        entity.get("end")
        if entity.get("end") is not None
        else entity.get("char_end")
    )

    return int(value)


# ============================================================
# FIND OCR LINE USING GLINER GLOBAL CHARACTER SPAN
# ============================================================

def find_ocr_line(
    entity_start,
    entity_end
):

    best_line = None

    best_overlap = 0


    for region in ocr_regions:

        overlap_start = max(
            entity_start,
            region["start"]
        )

        overlap_end = min(
            entity_end,
            region["end"]
        )

        overlap = max(
            0,
            overlap_end - overlap_start
        )


        if overlap > best_overlap:

            best_overlap = overlap

            best_line = region


    return best_line


# ============================================================
# FIND WORDS USING GLINER CHARACTER SPAN
# ============================================================

def find_matching_words(
    entity_start,
    entity_end,
    region
):

    matched_words = []


    # --------------------------------------------------------
    # Convert GLiNER global coordinates
    # to coordinates relative to this OCR line.
    # --------------------------------------------------------

    local_start = (
        entity_start
        - region["start"]
    )

    local_end = (
        entity_end
        - region["start"]
    )


    for word in region["words"]:

        word_start = word["start"]

        word_end = word["end"]


        if word_start is None:
            continue

        if word_end is None:
            continue


        # ----------------------------------------------------
        # Character-span intersection
        # ----------------------------------------------------

        if (
            word_start < local_end
            and word_end > local_start
        ):

            matched_words.append(
                word
            )


    return matched_words


# ============================================================
# UNION WORD BOXES
# ============================================================

def union_boxes(
    word_data
):

    if not word_data:

        return None


    xs1 = []
    ys1 = []
    xs2 = []
    ys2 = []


    for word in word_data:

        box = word["bbox"]

        xs1.append(
            float(box[0])
        )

        ys1.append(
            float(box[1])
        )

        xs2.append(
            float(box[2])
        )

        ys2.append(
            float(box[3])
        )


    return [

        min(xs1),

        min(ys1),

        max(xs2),

        max(ys2)

    ]


# ============================================================
# MAP PII TO EXACT WORD BOXES
# ============================================================

mapped_entities = []

unmapped_entities = []


print(
    "\nMapping PII entities "
    "to WORD-LEVEL OCR..."
)


for entity in entities:

    entity_text = get_entity_text(
        entity
    )

    entity_label = get_entity_label(
        entity
    )

    entity_score = get_entity_score(
        entity
    )

    entity_start = get_entity_start(
        entity
    )

    entity_end = get_entity_end(
        entity
    )


    # --------------------------------------------------------
    # Find OCR line
    # --------------------------------------------------------

    region = find_ocr_line(
        entity_start,
        entity_end
    )


    if region is None:

        unmapped_entities.append(
            entity
        )

        continue


    # --------------------------------------------------------
    # Find actual OCR words
    # overlapping the GLiNER span.
    # --------------------------------------------------------

    matched_words = (
        find_matching_words(
            entity_start,
            entity_end,
            region
        )
    )


    # --------------------------------------------------------
    # Exact PII bounding box
    # --------------------------------------------------------

    pii_bbox = union_boxes(
        matched_words
    )


    if pii_bbox is None:

        unmapped_entities.append(
            entity
        )

        continue


    # --------------------------------------------------------
    # Save complete mapping
    # --------------------------------------------------------

    mapped_entities.append({

        "text": entity_text,

        "type": entity_label,

        "confidence": entity_score,

        "gliner_start": entity_start,

        "gliner_end": entity_end,

        "ocr_region_index": (
            region["index"]
        ),

        "ocr_region_text": (
            region["text"]
        ),

        "ocr_region_bbox": (
            region["bbox"]
        ),

        "matched_words": [

            word["text"]

            for word in matched_words

        ],

        "word_boxes": [

            word["bbox"]

            for word in matched_words

        ],

        "pii_bbox": pii_bbox

    })


print(
    "Successfully mapped with "
    f"word-level boxes: "
    f"{len(mapped_entities)}"
)

print(
    "Unmapped entities: "
    f"{len(unmapped_entities)}"
)


# ============================================================
# SAVE JSON
# ============================================================

json_path = os.path.join(
    OUTPUT_DIR,
    "mapped_pii.json"
)


with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        mapped_entities,
        f,
        indent=4
    )


print(
    f"\nMapping saved to: "
    f"{json_path}"
)


# ============================================================
# VISUALIZATION
# ============================================================

print(
    "\nCreating precise PII visualization..."
)


image = Image.open(
    IMAGE_PATH
).convert("RGB")


draw = ImageDraw.Draw(
    image
)


# ============================================================
# FONT
# ============================================================

try:

    font = ImageFont.truetype(
        "arial.ttf",
        14
    )

except:

    font = ImageFont.load_default()


# ============================================================
# DRAW EACH PII ENTITY
# ============================================================

for item in mapped_entities:

    bbox = item["pii_bbox"]


    if bbox is None:

        continue


    x1, y1, x2, y2 = [

        int(round(v))

        for v in bbox

    ]


    # --------------------------------------------------------
    # Draw precise PII rectangle.
    # --------------------------------------------------------

    draw.rectangle(

        [x1, y1, x2, y2],

        outline="red",

        width=3

    )


    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label = (

        f'{item["text"]} '

        f'[{item["type"]}]'

    )


    # Measure label

    try:

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

    except:

        label_width = len(label) * 7

        label_height = 14


    # --------------------------------------------------------
    # Keep label inside image.
    # --------------------------------------------------------

    label_x = x1

    label_y = y1 - label_height - 3


    if label_y < 0:

        label_y = y2 + 3


    if (
        label_x
        + label_width
        > image.width
    ):

        label_x = max(
            0,
            image.width
            - label_width
        )


    # --------------------------------------------------------
    # Label background
    # --------------------------------------------------------

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


# ============================================================
# SAVE VISUALIZATION
# ============================================================

visualization_path = os.path.join(
    OUTPUT_DIR,
    "pii_word_level_visualization.png"
)


image.save(
    visualization_path
)


print(
    f"Visualization saved to: "
    f"{visualization_path}"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "PRECISE PII → WORD BOX MAPPING"
)

print(
    "=" * 70
)


for item in mapped_entities[:30]:

    print(
        "\nPII:",
        item["text"]
    )

    print(
        "Type:",
        item["type"]
    )

    print(
        "Confidence:",
        f'{item["confidence"]:.4f}'
    )

    print(
        "Matched words:",
        item["matched_words"]
    )

    print(
        "Precise PII bbox:",
        item["pii_bbox"]
    )


print(
    "\n"
    + "=" * 70
)

print(
    "STAGE 1 PRECISION MAPPING FINISHED"
)

print(
    "=" * 70
)