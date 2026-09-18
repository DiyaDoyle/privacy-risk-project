from paddleocr import PaddleOCR

IMAGE_PATH = "../sample.png"

print("=" * 60)
print("WORD-LEVEL OCR INSPECTION")
print("=" * 60)

print("\nLoading PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    return_word_box=True
)

print("PaddleOCR loaded successfully.")

print("\nRunning OCR...")

results = ocr.predict(
    input=IMAGE_PATH
)

for result in results:

    words = result["text_word"]
    boxes = result["text_word_boxes"]

    print("\n" + "=" * 60)
    print("WORD-LEVEL OCR RESULTS")
    print("=" * 60)

    print("\nNumber of words:", len(words))
    print("Number of word boxes:", len(boxes))

    print("\nFirst 50 words and their boxes:")

    for i, (word, box) in enumerate(zip(words, boxes)):

        if i >= 50:
            break

        print(f"\n{i}: {word}")
        print(f"   BOX: {box}")

print("\n" + "=" * 60)
print("WORD-LEVEL OCR INSPECTION COMPLETED")
print("=" * 60)