import re
import sys
import os
from tqdm import tqdm

def clean_log_file(input_path, output_path):
    """
    Reads a raw log file, cleans it according to predefined rules,
    and writes the result to an output file.

    This script performs several cleaning operations:
    1. Simplifies network function names in headers by removing unique IDs.
    2. Handles two distinct log formats:
       a) For logs with color codes, it removes all metadata BEFORE the log
          level keyword (e.g., INFO, WARNING).
       b) For logs starting with a bracketed timestamp, it removes only the
          timestamp.
    3. It removes all ANSI color/escape codes from any line.
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
    
    # Rule 2: A robust regex to find and remove ANSI escape codes (for colors, etc.).
    ansi_escape_regex = re.compile(r'(\x1b|)\[[0-9;]*m')

    # Rule 3: Finds the log line starting from a keyword (INFO, etc.) for the first log format.
    log_keywords = r'(?:INFO|WARNING|DEBUG|ERROR|CRITICAL|info|warning|error|debug|critical)' # Added lowercase
    log_line_finder = re.compile(f'({log_keywords}.*$)')

    # Rule 4: Matches and removes UUIDs or similar unique identifiers in brackets.
    uuid_regex = re.compile(
        r'\s*\[[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}(?::\d+)?\]'
    )
    
    # Rule 5: Matches and removes the trailing file path and line number info.
    file_path_regex = re.compile(r'\s*\([^)]+\)$')

    # Rule 6 (NEW): Specifically matches the timestamp prefix from the second log format.
    # e.g., '[2025-09-30 23:06:18.441]'
    timestamp_prefix_regex = re.compile(r'^\[\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d+\]\s*')
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
            cleaned_lines.append(f"{nf_match.group('name')} logs:")
            continue

        # --- REVISED LOGIC FOR CLEANING LOG MESSAGES ---
        message = None
        
        # Check for the new timestamp format first (Rule 6).
        if timestamp_prefix_regex.match(stripped_line):
            message = timestamp_prefix_regex.sub('', stripped_line)
        
        # Else, check for the original log format with keywords (Rule 3).
        else:
            log_match = log_line_finder.search(stripped_line)
            if log_match:
                message = log_match.group(1)
        
        # If a message was extracted by either rule, clean it further.
        if message is not None:
            # 1. Remove all ANSI codes (Rule 2).
            message = ansi_escape_regex.sub('', message)
            
            # 2. Remove the trailing file path info (Rule 5).
            message = file_path_regex.sub('', message)
            
            # 3. Remove any embedded UUIDs/IDs (Rule 4).
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
    input_dir = "/home/sharedrive/nafi/log_faults/dirty_dataset/network"
    output_dir = "/home/sharedrive/nafi/log_faults/clean_dataset/network"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of TXT files to process
    txt_files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".txt") and os.path.isfile(os.path.join(input_dir, f))
    ]

    for filename in tqdm(txt_files, desc="Processing DOCX files"):
        input_path = os.path.join(input_dir, filename)
        # extracting file_name without ext
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.txt"
        output_path = os.path.join(output_dir, output_filename)
        print(f"{filename} processing", flush=True)
        # Skip existing processed files
        if os.path.exists(output_path):
            print(f"{filename} already processed", flush=True)
            continue  # Silent skip (already shown in progress bar)
        
        clean_log_file(input_path, output_path)

    print("Processing complete")
