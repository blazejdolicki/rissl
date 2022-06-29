import argparse
import os
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, help="Directory with logs: checkpoints, parameters, metrics")
args = parser.parse_args()


metrics_path = os.path.join(args.log_dir, "metrics.json")
metrics = {"train": [],
           "valid": []}
with open(metrics_path) as json_file:
    phase_metrics = list(json_file)
    for train_phase_idx in range(0, len(phase_metrics), 2):
        epoch = int(train_phase_idx / 2)
        valid_phase_idx = train_phase_idx + 1

        train_acc = json.loads(phase_metrics[train_phase_idx])["train_accuracy_list_meter"]["top_1"]["0"]
        valid_acc = json.loads(phase_metrics[valid_phase_idx])["test_accuracy_list_meter"]["top_1"]["0"]

        metrics["train"].append(train_acc)
        metrics["valid"].append(valid_acc)

    print("metrics train", metrics["train"])
    print("metrics valid", metrics["valid"])

metrics_summmary = {"best_valid_acc": np.max(metrics["valid"]),
                    "best_valid_epoch": np.argmax(metrics["valid"]),
                    "final_valid_acc": metrics["valid"][-1],
                    }

losses_path = os.path.join(args.log_dir, "acc_summary.json")
with open(losses_path, 'w') as file:
    json.dump(metrics_summmary, file, indent=4)