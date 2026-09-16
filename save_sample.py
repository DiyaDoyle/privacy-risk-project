from datasets import load_dataset

# Load DocLayNet without downloading the whole dataset
ds = load_dataset(
    "docling-project/DocLayNet-v1.1",
    split="train",
    streaming=True
)

# Get one document page
sample = next(iter(ds))

# Save the image into our project
sample["image"].save("sample.png")

print("Sample saved successfully!")
print("Image size:", sample["image"].size)