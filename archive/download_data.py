from datasets import load_dataset

dataset = load_dataset("pierreguillou/DocLayNet-base", trust_remote_code=True)
print(dataset)
print(dataset['train'][0])