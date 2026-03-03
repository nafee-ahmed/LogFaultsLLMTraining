import os
import subprocess
import time
from datetime import datetime
import re

# ===== Configurable Variables =====
NAMESPACE = "open5gs"
POD_NAME = "nrf"
MAX_VARIATIONS = 100

# Directory to store the logs
POD_KILL_DIR = os.path.join(os.getcwd(), "pod_kill_logs")
os.makedirs(POD_KILL_DIR, exist_ok=True)

# WARNING: Setting this too low (e.g., 0.005) will overwhelm the
# Kubernetes API. A kubectl command itself takes time to run.
# 0.5 - 1.0 seconds is a reasonable starting point.
LOG_CHECK_INTERVAL = 0.5  # seconds between checks
BATCH_SIZE = 5  # number of new lines per log batch
YAML_FILE = "schedule-kill.yaml"  # The Chaos Mesh YAML file for pod-kill
VARIATION_NO = 1  # The script will increment this after each full loop

# This selector finds pods where the 'app' label is *either* 'open5gs' *or* 'ueransim'.
NF_LABEL_SELECTOR = "app in (open5gs, ueransim)"

# --- New Loop Configuration ---
CLEANUP_SCRIPT = "/home/k8s/open5gs-k8s/remove-all.sh"
DEPLOY_SCRIPT = "/home/k8s/open5gs-k8s/deploy-all.sh"

# How many log batches to collect *after* the pod kill is first detected
# This ensures the monitoring loop has a defined end.
LOG_BATCHES_POST_FAULT = 10
# Max time to wait for pods to start or stop
MAX_WAIT_TIME_SECS = 300
# ===== End Configurable Variables =====


def apply_yaml(yaml_file):
    """
    Applies a Kubernetes YAML file to the cluster.
    """
    if not os.path.exists(yaml_file):
        print(f"[ERROR] YAML file not found: {yaml_file}")
        print("[INFO] Skipping YAML application.")
        return False
    print(f"[INFO] Applying Kubernetes YAML: {yaml_file}")
    try:
        subprocess.run(
            ["kubectl", "apply", "-f", yaml_file, "-n", NAMESPACE], check=True
        )
        print(f"[INFO] Successfully applied {yaml_file}.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to apply YAML file {yaml_file}: {e}")
    except FileNotFoundError:
        print("[ERROR] `kubectl` command not found. Please ensure it's in your PATH.")
    return False


def delete_yaml(yaml_file):
    """
    Deletes a Kubernetes YAML file from the cluster.
    """
    if not os.path.exists(yaml_file):
        print(f"[ERROR] YAML file not found: {yaml_file}")
        print("[INFO] Skipping YAML deletion.")
        return False
    print(f"[INFO] Deleting Kubernetes YAML: {yaml_file}")
    try:
        # Use --ignore-not-found to avoid errors if the object is already gone
        subprocess.run(
            [
                "kubectl",
                "delete",
                "-f",
                yaml_file,
                "-n",
                NAMESPACE,
                "--ignore-not-found=true",
            ],
            check=True,
        )
        print(f"[INFO] Successfully deleted {yaml_file}.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to delete YAML file {yaml_file}: {e}")
    except FileNotFoundError:
        print("[ERROR] `kubectl` command not found.")
    return False


def run_script(script_path):
    """
    Executes a shell script (e.g., ./deploy-all.sh).

    FIXED:
    1. Removed capture_output to prevent pipe buffer deadlock with chatty scripts.
    2. Added the 'cwd' argument to set the working directory to the script's
       parent directory. This ensures relative paths (like 'chaos-mesh/mongodb')
       within the shell script are resolved correctly.
    """
    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        return False

    # --- FIX 1: Determine the correct working directory ---
    script_dir = os.path.dirname(script_path)
    # If the script path is just a file name, use the current directory
    if not script_dir:
        script_dir = os.getcwd()

    print(f"[INFO] Executing script: {script_path} from CWD: {script_dir}")
    print(f"-------------------- SCRIPT OUTPUT START --------------------")
    try:
        # Ensure the script is executable
        os.chmod(script_path, 0o755)

        # --- FIX 2 & 3: Run the script without capturing output and set CWD ---
        subprocess.run(
            [script_path],
            check=True,
            timeout=MAX_WAIT_TIME_SECS,
            # This is the CWD fix for relative paths
            cwd=script_dir,
        )

        print(f"--------------------- SCRIPT OUTPUT END ---------------------")
        print(f"[INFO] Successfully executed {script_path}.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"--------------------- SCRIPT OUTPUT END ---------------------")
        print(f"[ERROR] Script {script_path} failed with return code {e.returncode}.")
        return False
    except subprocess.TimeoutExpired:
        print(f"--------------------- SCRIPT OUTPUT END ---------------------")
        print(
            f"[ERROR] Script {script_path} timed out after {MAX_WAIT_TIME_SECS} seconds."
        )
        return False
    except FileNotFoundError:
        print(
            f"[ERROR] Could not execute script '{script_path}'. Is it in the correct directory?"
        )
        return False
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while running {script_path}: {e}")
        return False


def wait_for_pods_gone(timeout=MAX_WAIT_TIME_SECS):
    """
    Waits until no pods matching the NF_LABEL_SELECTOR are found.
    Verifies that the cleanup script worked.
    """
    print(f"[INFO] Waiting for all pods ({NF_LABEL_SELECTOR}) to be terminated...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        pod_map = get_nf_pod_map()
        if not pod_map:
            print("[INFO] All pods are gone.")
            return True
        print(f"[INFO] ... still waiting for {len(pod_map)} pod(s) to terminate.")
        time.sleep(LOG_CHECK_INTERVAL * 4)  # Check less frequently
    print(f"[ERROR] Timeout: Waited {timeout}s, but pods are still present.")
    return False


def wait_for_pods_running(timeout=MAX_WAIT_TIME_SECS):
    """
    Waits until at least one pod is found and ALL found pods are 'Running'.
    Verifies that the deploy script worked.
    """
    print(f"[INFO] Waiting for all pods ({NF_LABEL_SELECTOR}) to be 'Running'...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            command = [
                "kubectl",
                "get",
                "pods",
                "-n",
                NAMESPACE,
                "-l",
                NF_LABEL_SELECTOR,
                "-o",
                "jsonpath={.items[*].status.phase}",
            ]
            result = subprocess.check_output(command, stderr=subprocess.PIPE)
            statuses = result.decode().strip().split()

            if not statuses:
                print("[INFO] ... no pods found yet. Waiting for deploy script.")
                time.sleep(LOG_CHECK_INTERVAL * 4)
                continue

            all_running = True
            running_count = 0
            pending_pods = []
            for status in statuses:
                if status != "Running":
                    all_running = False
                    pending_pods.append(status)
                else:
                    running_count += 1

            if all_running:
                print(f"[INFO] All {running_count} found pods are 'Running'.")
                return True
            else:
                print(
                    f"[INFO] ... {running_count}/{len(statuses)} pods are Running. Waiting for: {pending_pods}"
                )

        except subprocess.CalledProcessError as e:
            # This can happen if kubectl fails, just retry
            print(
                f"[WARNING] `kubectl get pods` failed, will retry: {e.stderr.decode()}"
            )
        except FileNotFoundError:
            print("[ERROR] `kubectl` command not found.")
            return False

        time.sleep(LOG_CHECK_INTERVAL * 4)  # Check less frequently

    print(f"[ERROR] Timeout: Waited {timeout}s, but not all pods are 'Running'.")
    return False


def get_nf_pod_map():
    """
    Gets a mapping of Network Function (NF) names to their current pod names.
    e.g., {'open5gs-smf1': 'pod-name-1', 'open5gs-smf2': 'pod-name-2'}
    """
    nf_pod_map = {}
    try:
        command = [
            "kubectl",
            "get",
            "pods",
            "-n",
            NAMESPACE,
            "-l",
            NF_LABEL_SELECTOR,
            "-o",
            "custom-columns=APP:.metadata.labels.app,NF:.metadata.labels.nf,COMPONENT:.metadata.labels.component,NAME_LABEL:.metadata.labels.name,POD_NAME:.metadata.name",
            "--no-headers",
        ]
        result = subprocess.check_output(command, stderr=subprocess.PIPE)
        output = result.decode().strip()

        if not output:
            return {}

        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 5:
                app_label = parts[0]
                nf_label = parts[1]
                component_label = parts[2]
                name_label = parts[3]  # e.g., 'smf1', 'ue1', or '<none>'
                pod_name = parts[4]  # e.g., 'open5gs-smf1-...'

                key_name = None

                if name_label != "<none>":
                    if app_label == "open5gs":
                        key_name = f"open5gs-{name_label}"  # e.g., open5gs-smf1
                    elif app_label == "ueransim":
                        key_name = f"ueransim-{name_label}"  # e.g., ueransim-ue1

                elif app_label == "open5gs" and nf_label != "<none>":
                    key_name = f"open5gs-{nf_label}"  # e.g., open5gs-amf

                elif app_label == "ueransim" and component_label != "<none>":
                    key_name = f"ueransim-{component_label}"  # e.g., ueransim-gnb

                if key_name and pod_name != "<none>":
                    nf_pod_map[key_name] = pod_name

        return nf_pod_map
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to get pods: {e.stderr.decode()}")
        return {}
    except FileNotFoundError:
        print("[ERROR] `kubectl` command not found.")
        return {}


def write_log_file(all_nfs, new_log_batches, batch_number):
    """
    Writes the collected log batches to a numbered text file.
    Uses the simple NF name (e.g., "open5gs-amf" or "open5gs-smf1") for the header.
    """
    # VARIATION_NO is read as a global here
    filename = os.path.join(
        POD_KILL_DIR,
        f"{POD_NAME}_pod_kill_logs_batch_{batch_number}_var{VARIATION_NO}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.txt",
    )
    print(
        f"[INFO] Writing log batch {batch_number} (Variation {VARIATION_NO}) to {filename}..."
    )
    try:
        with open(filename, "w") as f:
            # Sort for consistent order in the file
            for nf_name in sorted(all_nfs):
                f.write(f"{nf_name} logs:\n")

                lines = new_log_batches.get(nf_name, [])

                for line in lines:
                    # Remove Kubernetes timestamp
                    line_no_stamp = re.sub(
                        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d+Z\s", "", line
                    )
                    f.write(line_no_stamp + "\n")
                f.write("\n\n")
        # print(f"[INFO] Successfully saved logs to {filename}") # Reduce log spam
    except IOError as e:
        print(f"[ERROR] Could not write to file {filename}: {e}")


# ===== Main Logic (Refactored) =====


def run_monitoring_cycle(baseline_line_counts, baseline_pod_names, yaml_file):
    """
    Applies the fault and monitors logs until LOG_BATCHES_POST_FAULT
    monitoring *loops* have been completed *after* the fault is detected.

    Args:
        baseline_line_counts (dict): {nf_name: line_count} from pre-fault
        baseline_pod_names (dict): {nf_name: pod_name} from pre-fault
        yaml_file (str): Path to the Chaos Mesh YAML
    """

    # We work on copies so the main loop's baseline isn't mutated
    saved_line_counts = baseline_line_counts.copy()
    tracked_pod_names = baseline_pod_names.copy()
    all_nfs_ever_tracked = set(tracked_pod_names.keys())  # Keep track of all NFs

    # ----- 1. Apply Fault -----
    if not apply_yaml(yaml_file):
        print("[ERROR] Failed to apply fault YAML. Aborting this cycle.")
        return
    print("[INFO] Fault applied. Baseline is set.")

    # ----- 2. Start Logging Loop -----
    print("[INFO] Starting monitoring loop... Monitoring for fault execution.")
    batch_number = 1
    fault_detected = False
    post_fault_loops_completed = 0  # <--- THIS IS THE KEY CHANGE

    try:
        while True:
            # 1. Get the current state
            current_nf_map = get_nf_pod_map()

            new_log_batches = {}
            any_new_logs = False

            # Check for *new* NFs that appeared post-baseline
            current_nf_set = set(current_nf_map.keys())
            tracked_nf_set = set(tracked_pod_names.keys())
            new_nfs = current_nf_set - tracked_nf_set
            for nf_name in new_nfs:
                pod_name = current_nf_map[nf_name]
                print(
                    f"[INFO] New NF detected (post-baseline): '{nf_name}' (Pod: {pod_name}). Adding to monitor list."
                )
                saved_line_counts[nf_name] = 0  # Start monitoring from line 0
                tracked_pod_names[nf_name] = pod_name
                all_nfs_ever_tracked.add(nf_name)  # Add to our master list

            # 2. Check for new logs on *all tracked* NFs
            for nf_name, last_known_pod in list(tracked_pod_names.items()):

                current_pod_name = current_nf_map.get(nf_name)
                last_saved_count = saved_line_counts.get(nf_name, 0)

                new_lines_from_old = []
                new_lines_from_new = []

                if current_pod_name != last_known_pod:
                    # ----- FAULT DETECTED (or pod restart) -----
                    if not fault_detected:
                        print(f"=====================================================")
                        print(f"[** FAULT DETECTED **] POD CHANGE FOR NF: '{nf_name}'")
                        print(f"[** FAULT DETECTED **]   Old Pod: {last_known_pod}")
                        print(f"[** FAULT DETECTED **]   New Pod: {current_pod_name}")
                        print(f"=====================================================")
                        fault_detected = True  # Start counting loops from now on

                    any_new_logs = True  # We must write this event

                    # Step A: Get final logs from the *old* pod
                    try:
                        old_logs_output = (
                            subprocess.check_output(
                                [
                                    "kubectl",
                                    "logs",
                                    "-n",
                                    NAMESPACE,
                                    last_known_pod,
                                    "--all-containers",
                                ],
                                stderr=subprocess.PIPE,
                            )
                            .decode()
                            .strip()
                        )
                        all_old_lines = old_logs_output.splitlines()
                        new_lines_from_old = all_old_lines[last_saved_count:]
                        print(
                            f"[INFO]   Captured {len(new_lines_from_old)} final lines from old pod {last_known_pod}."
                        )
                    except subprocess.CalledProcessError:
                        print(
                            f"[WARNING]     Could not get final logs for old pod {last_known_pod}."
                        )

                    # Step B: Get initial logs from the *new* pod
                    if current_pod_name:
                        try:
                            new_logs_output = (
                                subprocess.check_output(
                                    [
                                        "kubectl",
                                        "logs",
                                        "-n",
                                        NAMESPACE,
                                        current_pod_name,
                                        "--all-containers",
                                    ],
                                    stderr=subprocess.PIPE,
                                )
                                .decode()
                                .strip()
                            )
                            new_lines_from_new = new_logs_output.splitlines()
                            print(
                                f"[INFO]   Captured {len(new_lines_from_new)} initial lines from new pod {current_pod_name}."
                            )
                        except subprocess.CalledProcessError:
                            print(
                                f"[WARNING]     Could not get initial logs for new pod {current_pod_name}."
                            )

                        # Combine, save, and update baseline
                        new_lines_to_process = new_lines_from_old + new_lines_from_new
                        batch_to_save = new_lines_to_process[:BATCH_SIZE]
                        new_log_batches[nf_name] = batch_to_save

                        lines_saved_from_new_pod = max(
                            0, len(batch_to_save) - len(new_lines_from_old)
                        )
                        saved_line_counts[nf_name] = lines_saved_from_new_pod
                        tracked_pod_names[nf_name] = (
                            current_pod_name  # Start tracking the new pod
                        )
                        print(
                            f"[INFO]   New baseline for '{nf_name}' is {saved_line_counts[nf_name]} lines (from new pod)."
                        )

                    else:
                        # Pod just disappeared, no replacement
                        print(
                            f"[WARNING]     NF '{nf_name}' disappeared without replacement."
                        )
                        new_log_batches[nf_name] = new_lines_from_old[
                            :BATCH_SIZE
                        ]  # Save final logs
                        if nf_name in tracked_pod_names:
                            del tracked_pod_names[nf_name]
                        if nf_name in saved_line_counts:
                            del saved_line_counts[nf_name]

                else:
                    # ----- STABLE POD (no change) -----
                    if not current_pod_name:
                        continue

                    try:
                        all_logs_output = (
                            subprocess.check_output(
                                [
                                    "kubectl",
                                    "logs",
                                    "-n",
                                    NAMESPACE,
                                    current_pod_name,
                                    "--all-containers",
                                ],
                                stderr=subprocess.PIPE,
                            )
                            .decode()
                            .strip()
                        )

                        all_lines = all_logs_output.splitlines()
                        total_lines = len(all_lines)

                        if total_lines > last_saved_count:
                            any_new_logs = True
                            new_lines_to_process = all_lines[last_saved_count:]
                            batch_to_save = new_lines_to_process[:BATCH_SIZE]
                            new_log_batches[nf_name] = batch_to_save
                            saved_line_counts[nf_name] = last_saved_count + len(
                                batch_to_save
                            )

                    except subprocess.CalledProcessError:
                        print(
                            f"[WARNING] Could not fetch logs for NF '{nf_name}' (Pod: {current_pod_name})."
                        )
                    except Exception as e:
                        print(
                            f"[ERROR] An unexpected error occurred while fetching logs for {nf_name}: {e}"
                        )

            # 3. Write logs if any
            if any_new_logs:
                # Write a file containing *all* NFs we've ever seen for consistency
                write_log_file(
                    list(all_nfs_ever_tracked), new_log_batches, batch_number
                )
                batch_number += 1
            else:
                if fault_detected:
                    print(f"[INFO] Waiting (no new logs detected)...")
                else:
                    print(f"[INFO] Pre-fault: monitoring for pod change...")

            # 4. Check Exit Condition (This is the crucial change)
            # We increment the loop counter *every time* after a fault is detected,
            # regardless of whether new logs were found.
            if fault_detected:
                post_fault_loops_completed += 1
                print(
                    f"[INFO] Post-fault monitoring loops completed: {post_fault_loops_completed}/{LOG_BATCHES_POST_FAULT}"
                )

                if post_fault_loops_completed >= LOG_BATCHES_POST_FAULT:
                    print(
                        f"[INFO] Completed {post_fault_loops_completed} monitoring loops. Ending cycle."
                    )
                    break  # THIS IS THE EXIT

            # 5. Wait
            time.sleep(LOG_CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n[INFO] Stop request received. Exiting monitoring cycle.")
        raise  # Re-raise to stop the main_experiment_loop
    finally:
        print("[INFO] Monitoring cycle finished.")


def main_experiment_loop():
    """
    Main automation loop that manages the entire experiment lifecycle.
    Verify -> Baseline -> Inject Fault -> Monitor -> Cleanup -> Redeploy
    """
    global VARIATION_NO
    try:
        while True:
            if VARIATION_NO > MAX_VARIATIONS:
                print(
                    f"\n[INFO] Maximum limit of {MAX_VARIATIONS} experiment variations reached. Exiting."
                )
                break
            print(
                f"\n================================================================="
            )
            print(f"========= STARTING EXPERIMENT VARIATION {VARIATION_NO} ==========")
            print(
                f"=================================================================\n"
            )

            # ----- 1. Verify/Wait for Pods to be Running -----
            print("[INFO] STEP 1/5: Verifying all pods are in 'Running' state...")
            if not wait_for_pods_running(MAX_WAIT_TIME_SECS):
                print("[ERROR] Pods did not become 'Running'. Cannot proceed.")
                break
            print("[INFO] STEP 1/5: Verification complete. All pods are running.")

            # Give pods a moment to settle after starting
            print("[INFO] Waiting 15s for network functions to stabilize...")
            time.sleep(15)

            # ----- 2. Establish PRE-FAULT Baseline -----
            print("\n[INFO] STEP 2/5: Establishing pre-fault log baseline...")
            saved_line_counts = {}
            tracked_pod_names = {}
            try:
                initial_nf_map = get_nf_pod_map()
                if not initial_nf_map:
                    print(
                        f"[ERROR] No pods found with label selector '{NF_LABEL_SELECTOR}'. Cannot proceed."
                    )
                    break

                for nf_name, pod_name in initial_nf_map.items():
                    tracked_pod_names[nf_name] = pod_name
                    try:
                        logs_output = (
                            subprocess.check_output(
                                [
                                    "kubectl",
                                    "logs",
                                    "-n",
                                    NAMESPACE,
                                    pod_name,
                                    "--all-containers",
                                ],
                                stderr=subprocess.PIPE,
                            )
                            .decode()
                            .strip()
                        )
                        saved_line_counts[nf_name] = len(logs_output.splitlines())
                        print(
                            f"[INFO] Pre-fault baseline: NF '{nf_name}' (Pod: {pod_name}) has {saved_line_counts[nf_name]} lines."
                        )
                    except subprocess.CalledProcessError:
                        saved_line_counts[nf_name] = 0
                        print(
                            f"[INFO] Pre-fault baseline: NF '{nf_name}' (Pod: {pod_name}) has 0 lines."
                        )
                print("[INFO] STEP 2/5: Baseline complete.")

            except Exception as e:
                print(f"[ERROR] Failed during pre-fault baseline: {e}")
                break

            # ----- 3. Run Monitoring Cycle (Includes Fault Injection) -----
            print("\n[INFO] STEP 3/5: Starting monitoring cycle (will apply fault)...")
            run_monitoring_cycle(saved_line_counts, tracked_pod_names, YAML_FILE)
            print("[INFO] STEP 3/5: Monitoring cycle complete.")

            # ----- 4. Cleanup Environment -----
            print("\n[INFO] STEP 4/5: Cleaning up environment...")
            print("[INFO] Deleting Chaos Mesh YAML...")
            delete_yaml(YAML_FILE)

            print(f"[INFO] Running cleanup script: {CLEANUP_SCRIPT}")
            if not run_script(CLEANUP_SCRIPT):
                print(
                    f"[ERROR] Cleanup script {CLEANUP_SCRIPT} failed. Stopping experiment."
                )
                break

            if not wait_for_pods_gone(MAX_WAIT_TIME_SECS):
                print(
                    f"[ERROR] Pods were not removed after cleanup script. Stopping experiment."
                )
                break
            print("[INFO] STEP 4/5: Cleanup complete.")

            # ----- 5. Redeploy Environment -----
            print("\n[INFO] STEP 5/5: Redeploying environment for next variation...")
            print(f"[INFO] Running deploy script: {DEPLOY_SCRIPT}")
            if not run_script(DEPLOY_SCRIPT):
                print(
                    f"[ERROR] Deploy script {DEPLOY_SCRIPT} failed. Stopping experiment."
                )
                break

            # Note: wait_for_pods_running will be called at the start of the *next* loop.
            print("[INFO] STEP 5/5: Deploy script finished.")

            # --- End of cycle ---
            print(
                f"\n================================================================="
            )
            print(f"========= COMPLETED EXPERIMENT VARIATION {VARIATION_NO} ==========")
            print(
                f"=================================================================\n"
            )
            VARIATION_NO += 1
            print(f"Waiting 10s before starting Variation {VARIATION_NO}...")
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n[INFO] Stop request received. Exiting main experiment loop.")
    finally:
        print("[INFO] Automation loop stopped.")
        print("[INFO] Deleting Chaos Mesh YAML as a final cleanup step...")
        delete_yaml(YAML_FILE)  # Ensure fault is deleted on exit
        print("[INFO] Done.")


if __name__ == "__main__":
    all_files_found = True
    for f in [YAML_FILE, CLEANUP_SCRIPT, DEPLOY_SCRIPT]:
        if not os.path.exists(f):
            print(f"[ERROR] Required file not found: {f}")
            all_files_found = False

    if not all_files_found:
        print(
            "[INFO] Please create the missing files or update the variables in the script."
        )
    else:
        main_experiment_loop()
