import re
import sys
import os
from tqdm import tqdm

def clean_log_file(input_path, output_path):
    """
    Reads a raw log file, cleans it according to predefined rules,
    and writes the result to an output file.

    This script performs several cleaning operations:
    1. Simplifies network function names in headers by removing unique IDs
       (e.g., 'open5gs-amf-xxxxx...' becomes 'open5gs-amf').
    2. For log lines, it removes all metadata BEFORE the log level keyword
       (e.g., INFO, WARNING).
    3. It removes all ANSI color/escape codes.
    4. It removes source code file paths and unique IDs (like UUIDs) from
       the message body to focus on the event text itself.
    """
    print(f"Reading from '{input_path}'...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            input_text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    cleaned_lines = []
    
    # --- REGULAR EXPRESSIONS ---
    # Rule 1: Matches and captures the base name of a network function header.
    nf_name_regex = re.compile(
        r'^(?P<name>(?:open5gs|ueransim)-[a-zA-Z0-9]+)-[a-zA-Z0-9-]+ logs:$'
    )
    
    # Rule 2 (NEW): A robust regex to find and remove ANSI escape codes (for colors, etc.).
    # This will be applied to log lines after isolating them.
    ansi_escape_regex = re.compile(r'(\x1b|)\[[0-9;]*m')

    # Rule 3 (NEW): Finds the log line starting from the keyword (INFO, etc.)
    # and captures it. This replaces the old, buggy prefix remover.
    log_keywords = r'(?:INFO|WARNING|DEBUG|ERROR|CRITICAL)'
    log_line_finder = re.compile(f'({log_keywords}.*$)')

    # Rule 4: Matches and removes UUIDs or similar unique identifiers in brackets.
    uuid_regex = re.compile(
        r'\s*\[[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}(?::\d+)?\]'
    )
    
    # Rule 5: Matches and removes the trailing file path and line number info.
    file_path_regex = re.compile(r'\s*\([^)]+\)$')
    # --- END REGEX ---

    # Process the input text line by line
    for line in input_text.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            # Add a single blank line for readability between sections.
            if cleaned_lines and cleaned_lines[-1] != '':
                cleaned_lines.append('')
            continue

        # Apply Rule 1: Clean the network function name if the line is a header.
        nf_match = nf_name_regex.match(stripped_line)
        if nf_match:
            # Reconstruct the header using the captured base name.
            cleaned_lines.append(f"{nf_match.group('name')} logs:")
            continue

        # --- REVISED LOGIC FOR CLEANING LOG MESSAGES ---
        # Use the log_line_finder to see if this is a log message line.
        log_match = log_line_finder.search(stripped_line)
        if log_match:
            # We found a log line. Start with the keyword and everything after it.
            message = log_match.group(1)
            
            # Now, apply cleaning steps sequentially to this extracted message.
            # 1. Remove all ANSI codes.
            message = ansi_escape_regex.sub('', message)
            
            # 2. Remove the trailing file path info.
            message = file_path_regex.sub('', message)
            
            # 3. Remove any embedded UUIDs/IDs.
            message = uuid_regex.sub('', message)
            
            # Only add the cleaned message if there's content left.
            if message.strip():
                cleaned_lines.append(message.strip())
            continue
        # --- END REVISED LOGIC ---

        # Fallback: For any line that doesn't match the other rules (e.g., a version string),
        # keep it as is, as it might be important context.
        cleaned_lines.append(stripped_line)
        
    # Join all the cleaned lines back into a single string.
    final_output = '\n'.join(cleaned_lines).strip()

    # Write the final string to the output file.
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_output)
        
    print(f"Successfully cleaned the log file. Output saved to '{output_path}'.")

if __name__ == '__main__':
    input_dir = "/home/sharedrive/nafi/log_analysis/dirty_dataset"
    output_dir = "/home/sharedrive/nafi/log_analysis/cleaned_dataset"

    for root, dirs, files in os.walk(input_dir):
        # Filter for .txt files in the current directory
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
