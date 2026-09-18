from paddleocr import PaddleOCR
from pathlib import Path
img = Path("evaluation/dataset/images").glob("*.png").__next__()
print("TEST IMAGE:", img)
ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    return_word_box=True,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
results = list(ocr.predict(input=str(img)))
print("NUMBER OF RESULTS:", len(results))
if results:
    r = results[0]
    print("\nRESULT TYPE:", type(r))
    try:
        print("\nRESULT KEYS:")
        print(list(r.keys()))
    except Exception as e:
        print("Could not get keys:", e)
    print("\nRESULT:")
    print(r)
    for key in [
        "rec_texts",
        "rec_boxes",
        "text_word",
        "text_word_boxes",
        "rec_scores",
    ]:
        try:
            value = r.get(key)
            print(f"\n--- {key} ---")
            print("TYPE:", type(value))
            print("LEN:", len(value) if value is not None else None)
            print("SAMPLE:", value[:3] if value is not None else None)
        except Exception as e:
            print(key, "ERROR:", e)
