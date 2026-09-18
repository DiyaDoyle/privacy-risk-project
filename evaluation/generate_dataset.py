from datasets import load_dataset
from faker import Faker
from PIL import Image, ImageDraw, ImageFont
import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

NUM_DOCUMENTS = 5

OUTPUT_IMAGE_DIR = "evaluation/images"
OUTPUT_GT_DIR = "evaluation/ground_truth"

DATASET_NAME = "docling-project/DocLayNet-v1.1"
DATASET_SPLIT = "train"

fake = Faker("en_US")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_GT_DIR, exist_ok=True)


# ============================================================
# LOAD FONT
# ============================================================

FONT_PATH = "C:/Windows/Fonts/arial.ttf"

try:
    TITLE_FONT = ImageFont.truetype(FONT_PATH, 32)
    TEXT_FONT = ImageFont.truetype(FONT_PATH, 24)
except Exception:
    print("Arial font not found. Using default PIL font.")
    TITLE_FONT = ImageFont.load_default()
    TEXT_FONT = ImageFont.load_default()


# ============================================================
# GENERATE FAKE PII
# ============================================================

def generate_pii():

    name = fake.name()

    email = fake.email()

    phone = "+91 " + fake.msisdn()[3:13]

    address = (
        f"{fake.building_number()} "
        f"{fake.street_name()}, "
        f"{fake.city()}, "
        f"Telangana"
    )

    dob = fake.date_of_birth(
        minimum_age=18,
        maximum_age=70
    ).strftime("%d %B %Y")

    passport = (
        "P" +
        str(fake.random_number(
            digits=7,
            fix_len=True
        ))
    )

    return [
        {
            "text": name,
            "label": "full_name"
        },
        {
            "text": email,
            "label": "email"
        },
        {
            "text": phone,
            "label": "phone_number"
        },
        {
            "text": address,
            "label": "address"
        },
        {
            "text": dob,
            "label": "date_of_birth"
        },
        {
            "text": passport,
            "label": "passport_number"
        }
    ]


# ============================================================
# DRAW PII PANEL
# ============================================================

def add_pii_panel(image, pii_items, document_number):

    width, height = image.size

    # Add extra space below the original document.
    panel_height = 520

    new_image = Image.new(
        "RGB",
        (width, height + panel_height),
        "white"
    )

    # Put original DocLayNet page at the top.
    new_image.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_image)

    panel_y = height

    # Separator line
    draw.line(
        (0, panel_y, width, panel_y),
        fill="black",
        width=3
    )

    # Title
    draw.text(
        (60, panel_y + 25),
        f"PERSONAL INFORMATION - DOCUMENT {document_number}",
        fill="black",
        font=TITLE_FONT
    )

    y = panel_y + 80

    # Decide how PII is arranged.
    # This gives us different spatial patterns.
    layout_type = document_number % 3

    if layout_type == 1:
        # Clustered: all PII close together.
        positions = [
            (60, y),
            (60, y + 60),
            (60, y + 120),
            (60, y + 180),
            (60, y + 240),
            (60, y + 300),
        ]

    elif layout_type == 2:
        # Two-column arrangement.
        positions = [
            (60, y),
            (width // 2, y),
            (60, y + 120),
            (width // 2, y + 120),
            (60, y + 240),
            (width // 2, y + 240),
        ]

    else:
        # More vertically scattered arrangement.
        positions = [
            (60, y),
            (width // 2, y + 55),
            (60, y + 130),
            (width // 2, y + 185),
            (60, y + 260),
            (width // 2, y + 315),
        ]

    field_names = {
        "full_name": "Name",
        "email": "Email",
        "phone_number": "Phone",
        "address": "Address",
        "date_of_birth": "Date of Birth",
        "passport_number": "Passport Number"
    }

    for item, position in zip(pii_items, positions):

        label = field_names[item["label"]]

        text = f"{label}: {item['text']}"

        draw.text(
            position,
            text,
            fill="black",
            font=TEXT_FONT
        )

    return new_image


# ============================================================
# MAIN DATASET GENERATION
# ============================================================

def main():

    print("=" * 70)
    print("DOCULAYNET + FAKER SYNTHETIC PII DATASET")
    print("=" * 70)

    print("\nLoading DocLayNet-v1.1...")

    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        streaming=True
    )

    print("DocLayNet loaded successfully.")

    print(f"\nGenerating {NUM_DOCUMENTS} documents...")

    iterator = iter(dataset)

    generated = 0

    while generated < NUM_DOCUMENTS:

        try:
            sample = next(iterator)

        except StopIteration:
            print("\nDocLayNet ended before enough documents were generated.")
            break

        except Exception as error:
            print("\nError while reading DocLayNet:")
            print(error)
            print("Stopping safely.")
            break

        # ----------------------------------------------------
        # Get original image
        # ----------------------------------------------------

        image = sample["image"]

        if image is None:
            print("Skipping sample because image is missing.")
            continue

        image = image.convert("RGB")

        # ----------------------------------------------------
        # Generate PII
        # ----------------------------------------------------

        pii_items = generate_pii()

        # ----------------------------------------------------
        # Add PII
        # ----------------------------------------------------

        document_number = generated + 1

        modified_image = add_pii_panel(
            image,
            pii_items,
            document_number
        )

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        image_filename = (
            f"doc_{document_number:03d}.png"
        )

        image_path = os.path.join(
            OUTPUT_IMAGE_DIR,
            image_filename
        )

        modified_image.save(image_path)

        # ----------------------------------------------------
        # Save ground truth
        # ----------------------------------------------------

        ground_truth = {
            "document_id": f"doc_{document_number:03d}",

            "source": {
                "dataset": DATASET_NAME,
                "split": DATASET_SPLIT,
                "original_filename":
                    sample["metadata"].get(
                        "original_filename",
                        "unknown"
                    ),
                "page_no":
                    sample["metadata"].get(
                        "page_no",
                        None
                    )
            },

            "pii": pii_items
        }

        ground_truth_filename = (
            f"doc_{document_number:03d}.json"
        )

        ground_truth_path = os.path.join(
            OUTPUT_GT_DIR,
            ground_truth_filename
        )

        with open(
            ground_truth_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                ground_truth,
                file,
                indent=4,
                ensure_ascii=False
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        generated += 1

        print(
            f"\nDocument {generated}/{NUM_DOCUMENTS} created"
        )

        print(
            f"  Image: {image_path}"
        )

        print(
            f"  Ground truth: {ground_truth_path}"
        )

        print(
            f"  Source: "
            f"{ground_truth['source']['original_filename']}"
        )

        print(
            f"  Page: "
            f"{ground_truth['source']['page_no']}"
        )

        print("  PII inserted:")

        for item in pii_items:
            print(
                f"    - {item['label']}: "
                f"{item['text']}"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETED")
    print("=" * 70)

    print(
        f"\nDocuments generated: {generated}"
    )

    print(
        f"Images saved in: {OUTPUT_IMAGE_DIR}"
    )

    print(
        f"Ground truth saved in: {OUTPUT_GT_DIR}"
    )

    print("\nNext step will be running OCR + GLiNER2")
    print("against these documents.")
    print("=" * 70)


if __name__ == "__main__":
    main()