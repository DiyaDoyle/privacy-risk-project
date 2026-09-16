from gliner2 import GLiNER2
import json

print("=" * 60)
print("GLINER2-PII TEST")
print("=" * 60)

print("\nLoading GLiNER2-PII model...")

model = GLiNER2.from_pretrained(
    "fastino/gliner2-privacy-filter-PII-multi"
)

print("GLiNER2-PII loaded successfully.")

# Load PII labels from configuration
with open("config/pii_labels.json", "r", encoding="utf-8") as file:
    labels = json.load(file)

print(f"\nLoaded {len(labels)} PII labels from configuration.")

text = """
John Smith works at ABC Corporation.
His email is john.smith@example.com.
His phone number is +1 555-123-4567.
His address is 123 Main Street, New York.
"""

print("\nRunning PII detection...")

result = model.extract_entities(
    text,
    labels,
    threshold=0.5,
    include_confidence=True,
    include_spans=True
)

print("\n" + "=" * 60)
print("PII RESULTS")
print("=" * 60)

print(result)

print("\n" + "=" * 60)
print("GLINER2-PII TEST COMPLETED")
print("=" * 60)