import re
import sys
import os
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


def clean_network_function_names(line):
    """Simplifies network function names (e.g., open5gs-amf-xxx -> open5gs-amf)."""
    match = NF_NAME_REGEX.match(line)
    if match:
        return f"{match.group('name')} logs:"
    return line


def remove_metadata(line):
    """Removes everything before the log level (INFO, DEBUG, etc.)."""
    match = LOG_LINE_FINDER.search(line)
    if match:
        return match.group(1)
    return line


def remove_ansi_colors(line):
    """Removes ANSI color and escape codes."""
    return ANSI_ESCAPE_REGEX.sub("", line)


def remove_paths_and_ids(line):
    """Removes source code file paths and UUIDs from the message."""
    line = FILE_PATH_REGEX.sub("", line)
    line = UUID_REGEX.sub("", line)
    return line


def clean_log_file(input_path, output_path):
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            input_text = f.read()
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return

    cleaned_lines = []

    for line in input_text.splitlines():
        processed_line = line.strip()

        if not processed_line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        # --- TOGGLE FEATURES HERE ---
        processed_line = clean_network_function_names(processed_line)
        # processed_line = remove_metadata(processed_line)
        processed_line = remove_ansi_colors(processed_line)
        # processed_line = remove_paths_and_ids(processed_line)
        # ----------------------------

        if processed_line.strip():
            cleaned_lines.append(processed_line.strip())

    final_output = "\n".join(cleaned_lines).strip()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output)


if __name__ == "__main__":
    input_dir = "/home/sharedrive/nafi/log_analysis/datasets/dirty_dataset"
    output_dir = "/home/sharedrive/nafi/log_analysis/datasets/dirty_datasetv1"

    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)

    for root, dirs, files in os.walk(input_dir):
        txt_files = [f for f in files if f.endswith(".txt")]
        if not txt_files:
            continue

        relative_path = os.path.relpath(root, input_dir)
        target_output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(target_output_subdir, exist_ok=True)

        print(f"\n--- Processing folder: {relative_path} ---")

        for filename in tqdm(txt_files, desc=f"Cleaning {relative_path}"):
            input_path = os.path.join(root, filename)
            output_path = os.path.join(target_output_subdir, filename)

            if os.path.exists(output_path):
                continue

            clean_log_file(input_path, output_path)

    print("\nAll folders processed successfully.")
