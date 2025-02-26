import json
import os
import argparse
import numpy as np

# TODO: pass: checkpoint_dir
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="logs",
                    help="Directory with logs: checkpoints, parameters, metrics")
parser.add_argument("--subfolder_results_path", type=str, default="results.json")
args = parser.parse_args()

job_id = args.log_dir.split("/")[-1]

avg_results = {}
output_path = os.path.join(args.log_dir, "avg_seed_results.json")

for seed_folder in os.listdir(args.log_dir):
    if os.path.isdir(os.path.join(args.log_dir, seed_folder)):
        seed_dir = os.path.join(args.log_dir, seed_folder)
        acc_path = os.path.join(seed_dir, "evaluate", job_id, args.subfolder_results_path)

        with open(acc_path) as json_file:
            results = json.load(json_file)

        for split in results:
            if split not in avg_results:
                avg_results[split] = {}

            for metric in results[split]:
                metric_plural = metric + "s" if metric != "loss" else metric + "es"
                if metric_plural not in avg_results[split]:
                    avg_results[split][metric_plural] = []
                avg_results[split][metric_plural].append(results[split][metric])

        mre_path = os.path.join(seed_dir, "mre")
        if os.path.exists(mre_path):
            mres = os.listdir(mre_path)
            for mre_n in os.listdir(mre_path):
                mre_n_path = os.path.join(mre_path, mre_n, job_id, "results.json")

                with open(mre_n_path) as json_file:
                    mre_results = json.load(json_file)

                split = "test"
                if mre_n+"s" not in avg_results[split]:
                    avg_results[split][mre_n+"s"] = []
                avg_results[split][mre_n+"s"].append(mre_results[mre_n])

for split in avg_results:
    for metric in results[split]:
        metric_plural = metric + "s" if metric != "loss" else metric + "es"
        # setting axis=0 is especially important for non-scalar metrics such as confusion matrix
        avg_results[split]["avg_" + metric] = np.mean(avg_results[split][metric_plural], axis=0)
        avg_results[split]["std_" + metric] = np.std(avg_results[split][metric_plural], axis=0)
        if isinstance(results[split][metric], list):
            avg_results[split]["avg_" + metric] = avg_results[split]["avg_" + metric].tolist()
            avg_results[split]["std_" + metric] = avg_results[split]["std_" + metric].tolist()


if os.path.exists(mre_path):
    for mre_n in mres:
        avg_results["test"]["avg_"+mre_n] = np.mean(avg_results["test"][mre_n+"s"])
        avg_results["test"]["std_"+mre_n] = np.std(avg_results["test"][mre_n+"s"])

with open(output_path, 'w') as file:
    json.dump(avg_results, file, indent=4)
