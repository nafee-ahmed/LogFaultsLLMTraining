import os
import re
from datetime import datetime
from collections import Counter, defaultdict
import pandas as pd

def preprocess_5g_logs(raw_text):
    """
    Converts raw 5G log data into a structured, calculation-free text prompt
    for fault classification (network, stress, pod_kill).
    """

    # --- 1. PARSING & DATA EXTRACTION ---
    nf_logs = defaultdict(list)
    current_nf = None

    # Regex to capture: Timestamp, Level, Message
    # Supports both Open5GS (09/29 01:57:28.641) and UERANSIM ([2025-09-30 23:21:07.503])
    log_pattern = re.compile(
        r"(\d{2}/\d{2} \d{2}:\d{2}:\d{2}\.\d{3}|\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\])[:\s]+(?:\[(\w+)\])?\s*(\w+)?[:\s]+(.*)"
    )

    lines = raw_text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Identify Network Function block
        if "logs:" in line:
            current_nf = line.replace("logs:", "").strip()
            continue

        if current_nf:
            match = log_pattern.search(line)
            if match:
                ts_raw, module, level, msg = match.groups()
                # Normalize level (UERANSIM uses lowercase [error], Open5GS uses ERROR)
                lvl = (level or module or "INFO").upper()

                # Normalize timestamp for calculation
                try:
                    if "/" in ts_raw:  # Open5GS
                        ts_obj = datetime.strptime(ts_raw, "%m/%d %H:%M:%S.%f")
                    else:  # UERANSIM
                        ts_obj = datetime.strptime(
                            ts_raw.strip("[]"), "%Y-%m-%d %H:%M:%S.%f"
                        )

                    nf_logs[current_nf].append(
                        {"timestamp": ts_obj, "level": lvl, "message": msg, "raw": line}
                    )
                except Exception:
                    continue

    # --- 2. DERIVED CALCULATIONS ---
    all_errors = []
    for nf, logs in nf_logs.items():
        for log in logs:
            if log["level"] in ["ERROR", "WARNING", "CRITICAL"]:
                all_errors.append((log["timestamp"], nf, log))

    all_errors.sort(key=lambda x: x[0])  # Chronological order

    # 🧭 Failure Sequence
    sequence = []
    seen_nfs = set()
    for _, nf, _ in all_errors:
        if nf not in seen_nfs:
            sequence.append(nf)
            seen_nfs.add(nf)

    # ⏱ Failure Timing
    if all_errors:
        earliest_ts = all_errors[0][0]
        latest_ts = all_errors[-1][0]
        duration = (latest_ts - earliest_ts).total_seconds()
        burst_label = "Sudden/Concentrated" if duration < 5 else "Sustained/Extended"
    else:
        earliest_ts, duration, burst_label = "N/A", 0, "None"

    # 📊 Statistics & Keyword Detection
    keywords = {
        "heartbeat missing": ["No heartbeat", "heartbeat failure"],
        "NF deregistered": ["de-registered", "NF_DEREGISTERED", "Not found"],
        "connection timer expired": ["Connection timer expired", "timeout"],
        "no UPF available": ["No UPF available", "PFCP de-associated"],
        "cell selection failure": ["Cell selection failure", "no cells in coverage"],
        "radio link failure": ["Radio link failure"],
        "PFCP timeout": ["No Reponse", "PFCP", "LOCAL No Reponse"],
    }

    nf_stats = {}
    found_keywords = defaultdict(int)

    for nf, logs in nf_logs.items():
        counts = Counter(l["level"] for l in logs)
        msg_text = " ".join([l["message"] for l in logs])

        detected_keys = []
        for key, patterns in keywords.items():
            if any(p.lower() in msg_text.lower() for p in patterns):
                detected_keys.append(key)
                found_keywords[key] += 1

        nf_stats[nf] = {
            "counts": dict(counts),
            "keywords": detected_keys,
            "total_logs": len(logs),
        }

    # 🔗 Causal & Chaos Indicators (Rules Engine)
    causal_hints = []
    if "open5gs-nrf" in sequence and "open5gs-amf" in sequence:
        causal_hints.append(
            "Control-plane deregistration (NRF) observed before AMF heartbeat failure."
        )

    chaos_symptoms = []
    if any("de-registered" in str(l) for l in all_errors):
        chaos_symptoms.append(
            "Sudden NF disappearance/deregistration (suggests Pod Kill)"
        )
    if (
        found_keywords["connection timer expired"] > 2
        or found_keywords["PFCP timeout"] > 0
    ):
        chaos_symptoms.append(
            "High retry/timeout patterns (suggests Stress or Network latency)"
        )
    if (
        found_keywords["radio link failure"] > 0
        or found_keywords["cell selection failure"] > 0
    ):
        chaos_symptoms.append(
            "User-plane connectivity loss (suggests Network/Radio fault)"
        )

    # --- 3. OUTPUT GENERATION ---
    prompt = "### Observations\n"
    for nf, stat in nf_stats.items():
        if stat["total_logs"] > 0:
            prompt += f"- **{nf}**: Logged {stat['counts'].get('ERROR', 0)} errors and {stat['counts'].get('WARNING', 0)} warnings. "
            if stat["keywords"]:
                prompt += f"Key events: {', '.join(stat['keywords'])}. "
            else:
                prompt += "No critical protocol keywords detected. "
            prompt += "\n"

    prompt += "\n### Derived Indicators\n"
    prompt += f"- **Failure sequence**: {' → '.join(sequence) if sequence else 'No failures detected'}\n"
    prompt += f"- **Timing**: Errors started at {earliest_ts}. Total disturbance duration: {duration} seconds ({burst_label}).\n"
    prompt += f"- **Causal Hints**: {'; '.join(causal_hints) if causal_hints else 'No clear cross-NF causality detected.'}\n"
    prompt += f"- **Chaos Symptoms**: {', '.join(chaos_symptoms) if chaos_symptoms else 'No distinct chaos signatures detected.'}\n"

    prompt += "\n### Constraints / Exclusions\n"
    exclusions = []
    if (
        not found_keywords["cell selection failure"]
        and not found_keywords["radio link failure"]
    ):
        exclusions.append("No UE-side radio errors observed")
    if not found_keywords["no UPF available"]:
        exclusions.append("No UPF association failures detected")
    if "open5gs-nrf" not in nf_logs or not any(
        "ERROR" in str(l) for l in nf_logs["open5gs-nrf"]
    ):
        exclusions.append("NRF remained stable")

    for ex in exclusions:
        prompt += f"- {ex}\n"

    return prompt


def main():
    INPUT_CSV = "./datasets/prepared/dirty_datasetv1/train.csv"
    OUTPUT_DIR = "./datasets/precomputed/dirty_datasetv1/train.csv"

    df = pd.read_csv(INPUT_CSV)

    print("Processing logs and generating prompts...")
    df['question'] = df["question"].apply(preprocess_5g_logs)

    df.to_csv(OUTPUT_DIR, index=False)

    print(f"Success! Processed data saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
