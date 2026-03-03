import os
import re
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

def preprocess_5g_logs(raw_text):
    """
    Converts raw 5G log data into a structured, factual summary.
    Refactored to include module tracking, health metrics, and volume anomalies.
    """

    # --- 1. PARSING & DATA EXTRACTION ---
    nf_logs = defaultdict(list)
    current_nf = None

    # Regex captures: Timestamp, Module, Level, and Message
    log_pattern = re.compile(
        r"(\d{2}/\d{2} \d{2}:\d{2}:\d{2}\.\d{3}|\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\])[:\s]+(?:\[(\w+)\])?\s*(\w+)?[:\s]+(.*)"
    )

    lines = raw_text.splitlines()
    for line in lines:
        line = line.strip()
        if not line or "logs:" in line:
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
                        "timestamp": ts_obj, 
                        "level": lvl, 
                        "module": mod, 
                        "message": msg
                    })
                except Exception:
                    continue

    # --- 2. STATISTICAL AGGREGATION ---
    all_errors = []
    nf_health = {}
    total_log_counts = []

    for nf, logs in nf_logs.items():
        if not logs: continue
        
        l_types = Counter(l["level"] for l in logs)
        l_modules = Counter(l["module"] for l in logs)
        err_count = l_types.get("ERROR", 0) + l_types.get("CRITICAL", 0)
        warn_count = l_types.get("WARNING", 0)
        
        duration = (logs[-1]["timestamp"] - logs[0]["timestamp"]).total_seconds()
        
        nf_health[nf] = {
            "error_pct": round((err_count / len(logs)) * 100, 2),
            "warn_pct": round((warn_count / len(logs)) * 100, 2),
            "duration_sec": round(duration, 2),
            "total": len(logs),
            "modules": list(l_modules.keys())
        }
        total_log_counts.append(len(logs))

        for log in logs:
            if log["level"] in ["ERROR", "WARNING", "CRITICAL"]:
                all_errors.append((log["timestamp"], nf, log))

    all_errors.sort(key=lambda x: x[0])

    # 📈 Log Volume Anomalies
    avg_vol = np.mean(total_log_counts) if total_log_counts else 0
    anomalies = []
    for nf, metrics in nf_health.items():
        if metrics["total"] > avg_vol * 2:
            anomalies.append(f"{nf} (High: {metrics['total']} logs)")
        elif metrics["total"] < avg_vol * 0.5:
            anomalies.append(f"{nf} (Low: {metrics['total']} logs)")

    # ⏱ Failure Sequence & Gaps
    sequence_data = []
    seen_nfs = {}
    for ts, nf, _ in all_errors:
        if nf not in seen_nfs:
            gap = (ts - list(seen_nfs.values())[-1]).total_seconds() if seen_nfs else 0
            sequence_data.append((nf, gap))
            seen_nfs[nf] = ts

    # 🕒 Temporal Proximity (Failures within 1s)
    prox_pairs = []
    for i in range(len(all_errors) - 1):
        t1, nf1, _ = all_errors[i]
        t2, nf2, _ = all_errors[i+1]
        if nf1 != nf2 and (t2 - t1).total_seconds() <= 1.0:
            pair = tuple(sorted([nf1, nf2]))
            if pair not in prox_pairs: prox_pairs.append(pair)

    # 🔍 Expanded Keyword Detection (Fixed "Response")
    keywords = {
        "heartbeat_failure": ["No heartbeat", "heartbeat failure", "keep-alive timeout"],
        "registration_fault": ["de-registered", "NF_DEREGISTERED", "registration rejected", "403 Forbidden"],
        "timer_expiration": ["Connection timer expired", "retransmission limit reached"],
        "resource_exhaustion": ["No UPF available", "PFCP de-associated", "memory allocation failed"],
        "radio_link_loss": ["Radio link failure", "cell selection failure", "no cells in coverage"],
        "protocol_timeout": ["No Response", "PFCP", "LOCAL No Response", "SCTP shutdown"],
    }

    # --- 3. OUTPUT GENERATION ---
    prompt = "### NF Health Metrics\n"
    for nf, h in nf_health.items():
        prompt += f"- **{nf}**: {h['error_pct']}% Errors, {h['warn_pct']}% Warnings. Active for {h['duration_sec']}s. Modules: {', '.join(h['modules'])}\n"

    prompt += "\n### Log Volume Anomalies\n"
    prompt += f"- {', '.join(anomalies) if anomalies else 'Volume distribution within normal bounds.'}\n"

    prompt += "\n### Failure Timing & Sequence\n"
    if sequence_data:
        seq_str = " → ".join([f"{nf} (+{gap}s)" for nf, gap in sequence_data])
        prompt += f"- **Sequence**: {seq_str}\n"
        
        duration = (all_errors[-1][0] - all_errors[0][0]).total_seconds()
        err_rate = len(all_errors) / duration if duration > 0 else len(all_errors)
        prompt += f"- **Metrics**: {len(all_errors)} issues over {round(duration, 2)}s. Error Frequency: {round(err_rate, 2)} events/sec.\n"
    else:
        prompt += "- No failure sequence detected.\n"

    prompt += "\n### Temporal Proximity (Co-occurring <1s)\n"
    if prox_pairs:
        prompt += f"- Groups: {', '.join([f'[{a} & {b}]' for a, b in prox_pairs])}\n"
    else:
        prompt += "- No high-velocity co-occurrences detected.\n"

    prompt += "\n### Protocol Event Summary\n"
    found_any = False
    for nf, logs in nf_logs.items():
        msg_text = " ".join([l["message"] for l in logs]).lower()
        detected = [k for k, patterns in keywords.items() if any(p.lower() in msg_text for p in patterns)]
        if detected:
            prompt += f"- **{nf}**: {', '.join(detected)}\n"
            found_any = True
    if not found_any: prompt += "- No specific protocol patterns detected.\n"

    return prompt

def main():
    # Use your defined paths
    INPUT_CSV = "./datasets/prepared/dirty_datasetv1/train.csv"
    OUTPUT_DIR = "./datasets/precomputed/dirty_datasetv2/train.csv"

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