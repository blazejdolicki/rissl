import json
import os
import argparse
import numpy as np

# TODO: pass: checkpoint_dir
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="logs",
                    help="Directory with logs: checkpoints, parameters, metrics")
args = parser.parse_args()

job_id = args.log_dir.split("/")[-1]

avg_results = {}
output_path = os.path.join(args.log_dir, "avg_seed_results.json")

for seed_folder in os.listdir(args.log_dir):
    seed_dir = os.path.join(args.log_dir, seed_folder)
    acc_path = os.path.join(seed_dir, "evaluate", job_id, "results.json")

    with open(acc_path) as json_file:
        results = json.load(json_file)

    for split in results:
        if split not in avg_results:
            avg_results[split] = {}
            avg_results[split]["accs"] = []
        avg_results[split]["accs"].append(results[split]["acc"])

    mre_path = os.path.join(seed_dir, "mre")
    if os.path.exists(mre_path):
        # TODO remember there might be MRE(N) for multiple N
        first_mre_child = next(mre_path.iterdir())
        assert next(first_mre_child.iterdir()) is None, "There should be only one child in directory"
        mre_path = first_mre_child / "results.json"

        with open(mre_path) as json_file:
            results = json.load(json_file)

        for split in results:
            avg_results[split]["mre"].append(results[split]["mre"])


for split in avg_results:
    avg_results[split]["avg_acc"] = np.mean(avg_results[split]["accs"])
    avg_results[split]["std_acc"] = np.std(avg_results[split]["accs"])

with open(output_path, 'w') as file:
    json.dump(avg_results, file, indent=4)
