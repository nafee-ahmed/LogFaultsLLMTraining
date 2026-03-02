from datasets import load_dataset, concatenate_datasets
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

SYSTEM_PROMPT = """
You are a networking expert. You reason internally and then provide the answer.
The reasoning is enclosed within <think> and </think> tags, and the final answer is written as \\boxed{} with the corresponding MCQ label (e.g., C1, C2, A, B).

Format example:
<think>reasoning here</think>\n\\boxed{C1}
"""

def analyze_dataset_lengths(dataset):
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-7B-Instruct")

    input_lengths = []
    output_lengths = []

    for sample in dataset:
        input_len = len(tokenizer.encode(
            SYSTEM_PROMPT + sample["question"], add_special_tokens=True))
        output_len = len(tokenizer.encode(
            sample["answer"], add_special_tokens=True))

        input_lengths.append(input_len)
        output_lengths.append(output_len)

    # statistics
    max_input_len = max(input_lengths)
    mean_input_len = int(np.mean(input_lengths))
    p95_input_len = int(np.percentile(input_lengths, 95))

    max_output_len = max(output_lengths)
    mean_output_len = int(np.mean(output_lengths))
    p95_output_len = int(np.percentile(output_lengths, 95))

    print("=== Dataset Token Analysis ===")
    print(
        f"Input (question) tokens: max={max_input_len}, mean={mean_input_len}, 95th percentile={p95_input_len}")
    print(
        f"Output (answer) tokens: max={max_output_len}, mean={mean_output_len}, 95th percentile={p95_output_len}")

if __name__ == "__main__":
    # train_telelogs = load_dataset("csv", data_files="dataset/zindi/train.csv", split="train")
    col_shuffle_telelogs = load_dataset("csv", data_files="datasets/combined/dirty_datasetv1.csv", split="train")

    print(len(col_shuffle_telelogs))
    analyze_dataset_lengths(col_shuffle_telelogs)