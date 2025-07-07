import yaml
import pandas as pd
import wandb
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
output_dir = Path("experiment_metrics_fully_parallel")
output_dir.mkdir(exist_ok=True)
MAX_WORKERS = 64  # Adjust based on your network and CPU core count

# --- Initialize API once ---
api = wandb.Api()


def process_experiment(experiment_name, run_ids, entity, project):
    """
    Fetches, concatenates, and saves all metrics for a single experiment.
    This function is designed to be run in a separate thread.
    """
    if not isinstance(run_ids, list):
        run_ids = [run_ids]  # Ensure it's a list for consistency

    output_file = output_dir / f"{experiment_name}.csv"
    
    needs_update = False
    if not output_file.exists():
        needs_update = True
        print(f"🚀 Starting experiment: {experiment_name} (No local file)")
    else:
        local_mtime_utc = datetime.fromtimestamp(output_file.stat().st_mtime, tz=timezone.utc)
        print(f"🔎 Checking for updates for experiment: {experiment_name}...")
        
        for run_id in run_ids:
            try:
                run_path = f"{entity}/{project}/{run_id}"
                run = api.run(run_path)
                
                # --- CORRECTED TIMESTAMP LOGIC ---
                # Find the best available timestamp, preferring the most recent activity indicator.
                # 'heartbeat_at' is the most reliable indicator of recent activity.
                if hasattr(run, 'heartbeat_at') and run.heartbeat_at:
                    timestamp_str = run.heartbeat_at
                elif hasattr(run, 'updated_at') and run.updated_at:
                    timestamp_str = run.updated_at
                else:
                    # 'created_at' is a guaranteed fallback.
                    timestamp_str = run.created_at
                
                # Convert the chosen timestamp string to a timezone-aware datetime object
                run_activity_utc = pd.to_datetime(timestamp_str).tz_convert('UTC')

                if run_activity_utc > local_mtime_utc:
                    print(f"   - Stale data detected for {experiment_name}. Run '{run_id}' is newer.")
                    needs_update = True
                    break 
            except Exception as e:
                print(f"   - WARNING: Could not check update status for run {run_id}: {e}")
                needs_update = True 
                break

    if not needs_update:
        print(f"✅ Skipping experiment: {experiment_name} (Local file is up-to-date)")
        return f"Up-to-date: {experiment_name}"

    all_metrics = []
    print(f"🚀 Starting experiment: {experiment_name}")

    # Fetch runs sequentially for this experiment to maintain order
    for run_id in run_ids:
        try:
            run_path = f"{entity}/{project}/{run_id}"
            run = api.run(run_path)

            # Fetch only the specific metrics needed
            history = run.scan_history()
            metrics_df = pd.DataFrame(history)
            all_metrics.append(metrics_df)
            print(f"  - [{experiment_name}] Fetched metrics for run: {run_id}")

        except Exception as e:
            print(f"  - [{experiment_name}] ERROR fetching run {run_id}: {e}")
            # Continue to the next run_id in the list
            continue

    if not all_metrics:
        print(f"  - [{experiment_name}] No data found. Skipping save.")
        return f"No data for {experiment_name}"

    # Concatenate all dataframes for the experiment
    final_metrics = pd.concat(all_metrics, ignore_index=True)

    # Save the combined metrics to a CSV file
    output_file = output_dir / f"{experiment_name}.csv"
    final_metrics.to_csv(output_file, index=False)
    
    print(f"✅ Finished experiment: {experiment_name} -> Saved to {output_file}")
    return f"Successfully saved {experiment_name}"


# --- Main Execution Block ---
if __name__ == "__main__":
    with open("exp-escaip.yaml", "r") as f:
        config = yaml.safe_load(f)

    entity = config["entity"]
    project = config["project"]
    experiments = config["experiments"]

    # Use a ThreadPoolExecutor to run 'process_experiment' in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a future for each experiment
        future_to_experiment = {
            executor.submit(process_experiment, name, ids, entity, project): name
            for name, ids in experiments.items()
        }

        # As each future completes, print its result
        for future in as_completed(future_to_experiment):
            exp_name = future_to_experiment[future]
            try:
                result = future.result()
                # The detailed print statements are now inside the function
            except Exception as exc:
                print(f"'{exp_name}' generated an exception: {exc}")

    print("\nAll experiments have been processed.")