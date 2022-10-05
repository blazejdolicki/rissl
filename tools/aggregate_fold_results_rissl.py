import json
import os
import argparse
import numpy as np
from collections import defaultdict
import mlflow
import copy

"""
This script is very similar to average_seeds.py and they could be merged if time permits.
"""


def setup_mlflow(args):
    job_id = args.log_dir.split("/")[-1]
    mlflow.set_tracking_uri(f"file:///{args.mlflow_dir}")
    mlflow.set_experiment(args.exp_name)
    mlflow.start_run(run_name=job_id)
    mlflow_args = copy.deepcopy(args)
    mlflow_args.transform = "See args.json"
    mlflow.log_params(vars(mlflow_args))


parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="logs",
                    help="Directory with logs: checkpoints, parameters, metrics")
parser.add_argument("--mlflow_dir", type=str, default="/project/bdolicki/mlflow_runs",
                        help="Directory with MLFlow logs")
parser.add_argument('--exp_name', type=str, default="Default",
                        help='MLFlow experiment folder where the results will be logged')
args = parser.parse_args()

setup_mlflow(args)

job_id = args.log_dir.split("/")[-1]

avg_results = {}
output_path = os.path.join(args.log_dir, "avg_fold_results.json")

avg_results = defaultdict(list)
for fold_folder in os.listdir(args.log_dir):
    fold_dir = os.path.join(args.log_dir, fold_folder)
    acc_path = os.path.join(fold_dir, "metrics_summary.json")

    with open(acc_path) as json_file:
        results = json.load(json_file)

    avg_results["accs"].append(results["best_valid_acc"])
    avg_results["epochs"].append(results["best_valid_epoch"])


avg_results["avg_acc"] = np.mean(avg_results["accs"])
avg_results["std_acc"] = np.std(avg_results["accs"])
avg_results["avg_best_epoch"] = np.mean(avg_results["epochs"])
avg_results["std_best_epoch"] = np.std(avg_results["epochs"])

print("avg results", avg_results)
mlflow.log_metric("avg_acc", avg_results["avg_acc"])
mlflow.log_metric("std_acc", avg_results["std_acc"])
mlflow.log_metric("avg_best_epoch", avg_results["avg_best_epoch"])
mlflow.log_metric("std_best_epoch", avg_results["std_best_epoch"])

with open(output_path, 'w') as file:
    json.dump(avg_results, file, indent=4)
