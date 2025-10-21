from pathlib import Path
from datasets import Dataset, DatasetDict
from datasets import ClassLabel

def load_texts_with_labels(base_dir="dataset"):
    """Load text files from subdirectories with folder names as labels."""
    data = {"text": [], "label": []}
    
    base_path = Path(base_dir)
    category_names = sorted([d.name for d in base_path.iterdir() if d.is_dir()])
    # Iterate through category folders
    for category_dir in sorted(base_path.iterdir()):
        if category_dir.is_dir():
            category = category_dir.name
            txt_files = list(category_dir.glob("*.txt"))
            
            for txt_file in txt_files:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if text:
                    data["text"].append(text)
                    data["label"].append(category)
    
    return data, category_names

data_dict, label_names = load_texts_with_labels("clean_dataset")
dataset = Dataset.from_dict(data_dict)
print(label_names)
split_dataset_dict = dataset.class_encode_column("label").train_test_split(
    test_size=0.15, 
    shuffle=True,
    stratify_by_column="label"
)

def decode_labels(example):
    example['label_string'] = split_dataset_dict['train'].features['label'].int2str(example['label'])
    return example

train_dataset = split_dataset_dict['train'].map(decode_labels)
test_dataset = split_dataset_dict['test'].map(decode_labels)

train_dataset = train_dataset.remove_columns('label').rename_column('label_string', 'label')
test_dataset = test_dataset.remove_columns('label').rename_column('label_string', 'label')

final_dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
})

output_dir = "hf_dataset"
final_dataset.save_to_disk(output_dir)