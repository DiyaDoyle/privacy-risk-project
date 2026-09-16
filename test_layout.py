from paddleocr import LayoutDetection
import os

IMAGE_PATH = "sample.png"
OUTPUT_DIR = "layout_output"

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError("sample.png not found")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PP-DocLayout TEST")
print("=" * 60)

print("\nLoading PP-DocLayout-L...")

model = LayoutDetection(
    model_name="PP-DocLayout-L",
    device="cpu",
    enable_mkldnn=False
)

print("PP-DocLayout-L loaded successfully.")

print("\nRunning layout detection...")

results = model.predict(
    input=IMAGE_PATH,
    batch_size=1
)

for result in results:

    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)

    result.print()

    print("\nSaving annotated image...")
    result.save_to_img(save_path=OUTPUT_DIR)

    print("Saving JSON...")
    result.save_to_json(save_path=OUTPUT_DIR)

print("\n" + "=" * 60)
print("PP-DocLayout TEST COMPLETED")
print("=" * 60)

print(f"\nCheck the '{OUTPUT_DIR}' folder.")