from paddleocr import PaddleOCR
import os

IMAGE_PATH = "sample.png"
OUTPUT_DIR = "ocr_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PADDLEOCR TEST")
print("=" * 60)

print("\nLoading PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False
)

print("PaddleOCR loaded successfully.")

print("\nRunning OCR...")

results = ocr.predict(
    input=IMAGE_PATH
)

for result in results:

    print("\n" + "=" * 60)
    print("OCR RESULTS")
    print("=" * 60)

    result.print()

    print("\nSaving OCR result...")

    result.save_to_json(
        save_path=OUTPUT_DIR
    )

    result.save_to_img(
        save_path=OUTPUT_DIR
    )

print("\n" + "=" * 60)
print("PADDLEOCR TEST COMPLETED")
print("=" * 60)

print(f"\nCheck the '{OUTPUT_DIR}' folder.")