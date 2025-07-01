import yaml
import pandas as pd
import wandb
from pathlib import Path

# Create a directory to save the metrics
output_dir = Path("experiment_metrics")
output_dir.mkdir(exist_ok=True)

# Load the YAML file with your experiment data
with open("exp-prot.yaml", "r") as f:
    config = yaml.safe_load(f)

entity = config["entity"]
project = config["project"]
experiments = config["experiments"]

# Initialize the wandb API
api = wandb.Api()

# Process each experiment
for experiment_name, run_ids in experiments.items():
    if not isinstance(run_ids, list):
        run_ids = [run_ids]  # Ensure run_ids is always a list

    all_metrics = []
    print(f"Processing experiment: {experiment_name}")

    for run_id in run_ids:
        try:
            # Construct the full run path
            run_path = f"{entity}/{project}/{run_id}"
            run = api.run(run_path)

            # scan_history() is more efficient for fetching all data
            metrics_df = pd.DataFrame(run.scan_history())
            all_metrics.append(metrics_df)
            print(f"  - Fetched metrics for run ID: {run_id}")

        except wandb.errors.CommError as e:
            print(f"  - Could not find run with ID {run_id}: {e}")
        except Exception as e:
            print(f"  - An unexpected error occurred for run ID {run_id}: {e}")


    if all_metrics:
        # Concatenate metrics if there are multiple runs for an experiment
        final_metrics = pd.concat(all_metrics, ignore_index=True)

        # Save the combined metrics to a CSV file
        output_file = output_dir / f"{experiment_name}.csv"
        final_metrics.to_csv(output_file, index=False)
        print(f"  -> Saved metrics to {output_file}\n")
    else:
        print(f"  -> No metrics found for experiment: {experiment_name}\n")