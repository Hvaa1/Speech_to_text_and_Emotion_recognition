from datasets import Dataset, concatenate_datasets

base = r"D:\HuggingFaceCache\datasets\NhutP___viet_speech\default\0.0.0\0f18688afa9b64ea515d3fdaad850999b86a0f52"

# Lấy 3 shard đầu để nhẹ hơn
files = [f"{base}/viet_speech-train-{i:05d}-of-00262.arrow" for i in range(3)]

datasets = [Dataset.from_file(f) for f in files]
ds = concatenate_datasets(datasets)

print(ds)
print(ds[0])