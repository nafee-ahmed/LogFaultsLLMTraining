import re
import os
import pandas as pd
from tqdm import tqdm

# --- REGULAR EXPRESSIONS ---
NF_NAME_REGEX = re.compile(
    r"^(?P<name>(?:open5gs|ueransim)-[a-zA-Z0-9]+)-[a-zA-Z0-9-]+ logs:$"
)
ANSI_ESCAPE_REGEX = re.compile(r"(\x1b|)\[[0-9;]*m")
LOG_KEYWORDS = r"(?:INFO|WARNING|DEBUG|ERROR|CRITICAL)"
LOG_LINE_FINDER = re.compile(f"({LOG_KEYWORDS}.*$)")
UUID_REGEX = re.compile(r"\s*\[[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}(?::\d+)?\]")
FILE_PATH_REGEX = re.compile(r"\s*\([^)]+\)$")

# --- CLEANING UTILITIES ---


def clean_text_blob(text):
    """Applies all regex cleaning rules to a single string."""
    if not isinstance(text, str):
        return text

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        processed_line = line.strip()
        if not processed_line:
            continue

        # 1. Simplify NF names
        match = NF_NAME_REGEX.match(processed_line)
        if match:
            processed_line = f"{match.group('name')} logs:"

        # 2. Remove metadata (keep only log level onwards)
        match_log = LOG_LINE_FINDER.search(processed_line)
        if match_log:
            processed_line = match_log.group(1)

        # 3. Remove ANSI/Colors
        processed_line = ANSI_ESCAPE_REGEX.sub("", processed_line)

        # 4. Remove Paths and UUIDs
        processed_line = FILE_PATH_REGEX.sub("", processed_line)
        processed_line = UUID_REGEX.sub("", processed_line)

        if processed_line.strip():
            cleaned_lines.append(processed_line.strip())

    return "\n".join(cleaned_lines).strip()


def process_csv_directory(input_dir, output_dir, file_name):
    """Reads CSV from input_dir, cleans 'question' and 'answer', saves to output_dir."""

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)

    if not os.path.exists(input_path):
        print(f"Error: File {file_name} not found in {input_dir}")
        return

    print(f"Processing {file_name}...")

    # Load CSV
    df = pd.read_csv(input_path)

    # Apply cleaning to target columns
    target_cols = ["question", "answer"]

    for col in target_cols:
        if col in df.columns:
            # Using tqdm to track progress across rows
            tqdm.pandas(desc=f"Cleaning {col}")
            df[col] = df[col].progress_apply(clean_text_blob)
        else:
            print(f"Warning: Column '{col}' not found in {file_name}")

    # Save output
    df.to_csv(output_path, index=False)
    print(f"Successfully saved cleaned file to: {output_path}")


# --- EXECUTION ---

if __name__ == "__main__":
    # Specify your directories here
    INPUT_DIRECTORY = "./datasets/prepared/dirty_dataset"
    OUTPUT_DIRECTORY = "./datasets/precomputed/cleaned_datasetv1"
    FILE_TO_PROCESS = "train.csv"

    process_csv_directory(INPUT_DIRECTORY, OUTPUT_DIRECTORY, FILE_TO_PROCESS)
