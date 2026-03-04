import os
import re
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

def preprocess_5g_logs(raw_text):
    """
    v4 Preprocessor: Features statistical evidence over raw logs.
    Focuses on frequency, temporal patterns, and subsystem failures.
    """

    # --- 1. MASTER LIST & KEYWORDS ---
    ALL_POSSIBLE_NFS = [
        "open5gs-amf", "open5gs-ausf", "open5gs-bsf", "open5gs-nrf",
        "open5gs-nssf", "open5gs-pcf", "open5gs-scp", "open5gs-smf1",
        "open5gs-smf2", "open5gs-udm", "open5gs-udr", "open5gs-upf1",
        "open5gs-upf2", "open5gs-webui", "ueransim-gnb", "ueransim-ue1", "ueransim-ue2"
    ]

    KEYWORDS = {
        "heartbeat_failure": ["No heartbeat", "heartbeat failure", "keep-alive timeout"],
        "registration_fault": ["de-registered", "NF_DEREGISTERED", "registration rejected", "403 Forbidden"],
        "timer_expiration": ["Connection timer expired", "retransmission limit reached"],
        "resource_exhaustion": ["No UPF available", "PFCP de-associated", "memory allocation failed"],
        "radio_link_loss": ["Radio link failure", "cell selection failure"],
        "protocol_timeout": ["No Response", "PFCP", "LOCAL No Response", "SCTP shutdown"],
    }

    # --- 2. PARSING ---
    nf_logs = defaultdict(list)
    current_nf = None
    log_pattern = re.compile(
        r"(\d{2}/\d{2} \d{2}:\d{2}:\d{2}\.\d{3}|\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\])[:\s]+(?:\[(\w+)\])?\s*(\w+)?[:\s]+(.*)"
    )

    lines = raw_text.splitlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        if "logs:" in line:
            current_nf = line.replace("logs:", "").strip()
            continue

        if current_nf:
            match = log_pattern.search(line)
            if match:
                ts_raw, module, level, msg = match.groups()
                lvl = (level or module or "INFO").upper()
                mod = (module if module else "UNKNOWN")

                try:
                    if "/" in ts_raw:
                        ts_obj = datetime.strptime(ts_raw, "%m/%d %H:%M:%S.%f")
                    else:
                        ts_obj = datetime.strptime(ts_raw.strip("[]"), "%Y-%m-%d %H:%M:%S.%f")

                    nf_logs[current_nf].append({
                        "timestamp": ts_obj, "level": lvl,
                        "module": mod, "message": msg
                    })
                except: continue

    # --- 3. CORE ANALYTICS ---
    all_errors = []
    total_log_counts = []

    for nf, logs in nf_logs.items():
        total_log_counts.append(len(logs))
        for log in logs:
            if log["level"] in ["ERROR", "CRITICAL"]:
                all_errors.append({"ts": log["timestamp"], "nf": nf, "data": log})

    all_errors.sort(key=lambda x: x["ts"])

    # A. Failure Sequence & Gaps
    sequence_data = []
    seen_nfs = set()
    for i, err in enumerate(all_errors):
        if err["nf"] not in seen_nfs:
            gap = (err["ts"] - all_errors[i-1]["ts"]).total_seconds() if i > 0 else 0
            sequence_data.append((err["nf"], gap))
            seen_nfs.add(err["nf"])

    # B. Temporal Pattern Classification
    pattern_desc = "None"
    avg_gap, variance = 0, 0
    if len(all_errors) > 1:
        gaps = [(all_errors[i]["ts"] - all_errors[i-1]["ts"]).total_seconds() for i in range(1, len(all_errors))]
        avg_gap, variance, std_dev = np.mean(gaps), np.var(gaps), np.std(gaps)
        if std_dev > avg_gap:
            pattern_desc = "Burst Pattern (Highly Clustered)"
        elif max(gaps) > avg_gap * 3:
            pattern_desc = "Multi-phase Pattern (Wave-like)"
        else:
            pattern_desc = "Uniform/Steady Pattern"

    # C. Synchronized Failures (<1s)
    prox_pairs = defaultdict(int)
    for i in range(len(all_errors)):
        for j in range(i + 1, len(all_errors)):
            diff = (all_errors[j]["ts"] - all_errors[i]["ts"]).total_seconds()
            if diff > 1.0: break
            if all_errors[i]["nf"] != all_errors[j]["nf"]:
                pair = tuple(sorted([all_errors[i]["nf"], all_errors[j]["nf"]]))
                prox_pairs[pair] += 1

    # --- 4. OUTPUT GENERATION ---
    prompt = "### NF Health & Volume\n"
    avg_vol = np.mean(total_log_counts) if total_log_counts else 0
    for nf in sorted(nf_logs.keys()):
        logs = nf_logs[nf]
        errs = [l for l in logs if l["level"] in ["ERROR", "CRITICAL"]]
        warns = [l for l in logs if l["level"] == "WARNING"]
        vol_stat = " (Anomaly: High)" if len(logs) > avg_vol*2 else " (Anomaly: Low)" if len(logs) < avg_vol*0.5 else ""
        prompt += f"- **{nf}**: {len(errs)} errors, {len(warns)} warns. Logs: {len(logs)}{vol_stat}\n"

    prompt += "\n### Error Temporal Pattern\n"
    prompt += f"- **Classification**: {pattern_desc}\n"
    prompt += f"- **Inter-error gap**: mean={round(avg_gap, 3)}s, var={round(variance, 3)}\n"

    prompt += "\n### Failure Sequence & Timing\n"
    if sequence_data:
        prompt += "- " + " → ".join([f"{nf} (+{gap}s)" for nf, gap in sequence_data]) + "\n"

    prompt += "\n### Synchronized Failures (<1s)\n"
    if prox_pairs:
        for (n1, n2), count in sorted(prox_pairs.items(), key=lambda x: -x[1])[:5]:
            prompt += f"- {n1} ↔ {n2}: {count} occurrences\n"
    else: prompt += "- No synchronized cross-NF bursts detected.\n"

    prompt += "\n### Sample Error Messages (Top Unique)\n"
    for nf, logs in nf_logs.items():
        err_msgs = [l["message"] for l in logs if l["level"] in ["ERROR", "CRITICAL"]]
        if err_msgs:
            top_3 = Counter(err_msgs).most_common(3)
            prompt += f"- **{nf}**:\n"
            for msg, count in top_3:
                prompt += f"  - \"{msg}\" ({count}x)\n"

    prompt += "\n### Protocol Signatures\n"
    for nf, logs in nf_logs.items():
        counts = defaultdict(int)
        for l in logs:
            msg = l["message"].lower()
            for key, patterns in KEYWORDS.items():
                if any(p.lower() in msg for p in patterns):
                    counts[key] += 1
        if counts:
            items = ", ".join([f"{k}({v}x)" for k, v in sorted(counts.items(), key=lambda x: -x[1])])
            prompt += f"- **{nf}**: {items}\n"

    prompt += "\n### Missing/Silent NFs\n"
    silent = [nf for nf in ALL_POSSIBLE_NFS if nf not in nf_logs]
    prompt += f"- {', '.join(silent) if silent else 'None'}\n"

    return prompt


def main():
    # Use your defined paths
    INPUT_CSV = "./datasets/prepared/dirty_datasetv1/train.csv"
    OUTPUT_DIR = "./datasets/precomputed/dirty_datasetv3/train.csv"

    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    print("Extracting factual features and calculating NF health...")
    df['question'] = df["question"].apply(preprocess_5g_logs)

    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    df.to_csv(OUTPUT_DIR, index=False)
    print(f"Success! Processed data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
