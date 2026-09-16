from gliner2 import GLiNER2
from paddleocr import PaddleOCR
import json
import os

print("=" * 60)
print("GLINER2 + PADDLEOCR INTEGRATION TEST")
print("=" * 60)

IMAGE_PATH = "sample.png"

# --------------------------------------------------
# 1. Load PII labels
# --------------------------------------------------

print("\nLoading PII labels...")

with open("config/pii_labels.json", "r", encoding="utf-8") as file:
    labels = json.load(file)

print(f"Loaded {len(labels)} PII labels.")

# --------------------------------------------------
# 2. Load PaddleOCR
# --------------------------------------------------

print("\nLoading PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False
)

print("PaddleOCR loaded successfully.")

# --------------------------------------------------
# 3. Run OCR
# --------------------------------------------------

print("\nRunning OCR on sample.png...")

ocr_results = ocr.predict(
    input=IMAGE_PATH
)

print("OCR completed.")

# --------------------------------------------------
# 4. Extract OCR text
# --------------------------------------------------

all_text = []

for result in ocr_results:
    if hasattr(result, "json"):
        data = result.json

        if isinstance(data, str):
            data = json.loads(data)

        if "res" in data:
            data = data["res"]

        texts = data.get("rec_texts", [])

        for text in texts:
            if text and text.strip():
                all_text.append(text.strip())

combined_text = "\n".join(all_text)

print(f"\nOCR extracted {len(all_text)} text regions.")

print("\nFirst OCR text:")
print(combined_text[:1000])

# --------------------------------------------------
# 5. Load GLiNER2-PII
# --------------------------------------------------

print("\nLoading GLiNER2-PII model...")

model = GLiNER2.from_pretrained(
    "fastino/gliner2-privacy-filter-PII-multi"
)

print("GLiNER2-PII loaded successfully.")

# --------------------------------------------------
# 6. Run PII detection on OCR text
# --------------------------------------------------

print("\nRunning GLiNER2-PII on OCR text...")

result = model.extract_entities(
    combined_text,
    labels,
    threshold=0.5,
    include_confidence=True,
    include_spans=True
)

# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("PII DETECTION RESULTS")
print("=" * 60)

print(result)

print("\n" + "=" * 60)
print("GLINER2 + PADDLEOCR TEST COMPLETED")
print("=" * 60)