from datasets import load_from_disk, load_dataset
from collections import Counter
import pandas as pd

dataset = load_from_disk("hf_dataset")

for split in dataset.keys():
    counts = Counter(dataset[split]['label'])
    print(f"\nCounts for [{split}] split:")
    print(counts)

dataset.push_to_hub(repo_id="nafi-ahmed/log_faults", private=True)