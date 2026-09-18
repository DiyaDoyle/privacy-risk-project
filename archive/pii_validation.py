import json
import os
import re

from paddleocr import PaddleOCR
from gliner2 import GLiNER2


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = "../sample.png"
LABEL_PATH = "../config/pii_labels.json"
OUTPUT_DIR = "../validation_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD PII LABELS
# ============================================================

with open(LABEL_PATH, "r", encoding="utf-8") as file:
    labels = json.load(file)

print(f"Loaded {len(labels)} PII labels from configuration.")


# ============================================================
# LOAD OCR
# ============================================================

print("\nLoading PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False
)

print("PaddleOCR loaded successfully.")


# ============================================================
# RUN OCR
# ============================================================

print("\nRunning OCR...")

ocr_results = ocr.predict(
    input=IMAGE_PATH
)

ocr_regions = []

for result in ocr_results:

    texts = result["rec_texts"]
    boxes = result["rec_boxes"]
    scores = result["rec_scores"]

    for text, box, score in zip(texts, boxes, scores):

        ocr_regions.append({
            "text": text,
            "bbox": box.tolist() if hasattr(box, "tolist") else box,
            "confidence": float(score)
        })


print(f"OCR regions collected: {len(ocr_regions)}")


# ============================================================
# COMBINE OCR TEXT
# ============================================================

combined_text = ""

for region in ocr_regions:
    combined_text += region["text"] + " "


combined_text = combined_text.strip()

print("\nCombined OCR text created.")


# ============================================================
# LOAD GLINER2-PII
# ============================================================

print("\nLoading GLiNER2-PII...")

model = GLiNER2.from_pretrained(
    "fastino/gliner2-privacy-filter-PII-multi"
)

print("GLiNER2-PII loaded successfully.")


# ============================================================
# RUN PII DETECTION
# ============================================================

print("\nRunning PII detection...")

raw_result = model.extract_entities(
    combined_text,
    labels,
    threshold=0.5,
    include_confidence=True,
    include_spans=True
)


# ============================================================
# FLATTEN GLINER OUTPUT
# ============================================================

entities = []

if isinstance(raw_result, dict) and "entities" in raw_result:

    for entity_type, entity_list in raw_result["entities"].items():

        for entity in entity_list:

            entities.append({
                "text": entity.get("text"),
                "label": entity_type,
                "confidence": entity.get("confidence"),
                "start": entity.get("start"),
                "end": entity.get("end")
            })


print(f"Raw PII detections: {len(entities)}")


# ============================================================
# VALIDATION RULES
# ============================================================

STRONG_PII_TYPES = {
    "person",
    "full_name",
    "first_name",
    "middle_name",
    "last_name",
    "email",
    "phone_number",
    "address",
    "street_address",

    "government_id",
    "national_id_number",
    "passport_number",
    "drivers_license_number",
    "license_number",
    "tax_id",
    "tax_number",

    "bank_account",
    "account_number",
    "routing_number",
    "iban",
    "payment_card",
    "card_number",
    "card_expiry",
    "card_cvv",

    "username",
    "ip_address",
    "account_id",
    "sensitive_account_id",

    "password",
    "secret",
    "api_key",
    "access_token",
    "recovery_code"
}


DATE_TYPES = {
    "date_of_birth",
    "sensitive_date",
    "document_date",
    "expiration_date",
    "transaction_date"
}


AMBIGUOUS_TYPES = {
    "country",
    "city",
    "state_or_region"
}


YEAR_PATTERN = re.compile(r"^\d{4}$")

MONEY_PATTERN = re.compile(
    r"^[\$₹€£]?\s?\d[\d,]*(\.\d+)?$"
)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate_entity(entity):

    text = str(entity["text"]).strip()
    label = entity["label"]
    confidence = float(entity["confidence"])

    # --------------------------------------------------------
    # Strong PII
    # --------------------------------------------------------

    if label in STRONG_PII_TYPES:

        if confidence >= 0.70:
            return True, "strong PII with sufficient confidence"

        return False, "strong PII below confidence threshold"


    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    if label in DATE_TYPES:

        # Ordinary four-digit years such as 2004 or 2005
        # should not automatically be treated as personal dates.

        if YEAR_PATTERN.match(text):

            return False, "ordinary four-digit year"

        if confidence >= 0.70:

            return True, "date with sufficient confidence"

        return False, "date below confidence threshold"


    # --------------------------------------------------------
    # Country / City / State
    # --------------------------------------------------------

    if label in AMBIGUOUS_TYPES:

        if confidence >= 0.70:

            return True, "location entity with sufficient confidence"

        return False, "ambiguous location with low confidence"


    # --------------------------------------------------------
    # Financial-document false positives
    # --------------------------------------------------------

    if "SFAS No." in text.upper():

        if label in {"tax_id", "tax_number"}:

            return False, "accounting standard reference, not tax identifier"


    # --------------------------------------------------------
    # Money values
    # --------------------------------------------------------

    if MONEY_PATTERN.match(text):

        return False, "financial amount, not PII"


    # --------------------------------------------------------
    # Default rule
    # --------------------------------------------------------

    if confidence >= 0.70:

        return True, "sufficient confidence"

    return False, "below confidence threshold"


# ============================================================
# VALIDATE ALL DETECTIONS
# ============================================================

validated = []
rejected = []

for entity in entities:

    is_valid, reason = validate_entity(entity)

    entity["validation_reason"] = reason

    if is_valid:
        validated.append(entity)

    else:
        rejected.append(entity)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    os.path.join(OUTPUT_DIR, "validated_pii.json"),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        validated,
        file,
        indent=4,
        ensure_ascii=False
    )


with open(
    os.path.join(OUTPUT_DIR, "rejected_pii.json"),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        rejected,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# VALIDATION REPORT
# ============================================================

report = {
    "image": IMAGE_PATH,
    "total_ocr_regions": len(ocr_regions),
    "raw_pii_detections": len(entities),
    "validated_pii": len(validated),
    "rejected_detections": len(rejected)
}


with open(
    os.path.join(OUTPUT_DIR, "validation_report.json"),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print(f"OCR regions:       {len(ocr_regions)}")
print(f"Raw PII:           {len(entities)}")
print(f"Validated PII:     {len(validated)}")
print(f"Rejected:          {len(rejected)}")


print("\n" + "=" * 60)
print("VALIDATED PII")
print("=" * 60)

for entity in validated:

    print(
        f"{entity['text']} | "
        f"{entity['label']} | "
        f"{entity['confidence']:.3f} | "
        f"{entity['validation_reason']}"
    )


print("\n" + "=" * 60)
print("REJECTED DETECTIONS")
print("=" * 60)

for entity in rejected:

    print(
        f"{entity['text']} | "
        f"{entity['label']} | "
        f"{entity['confidence']:.3f} | "
        f"{entity['validation_reason']}"
    )


print("\n" + "=" * 60)
print("STAGE 2 COMPLETED")
print("=" * 60)

print("\nFiles saved in:")
print(f"  {OUTPUT_DIR}/")