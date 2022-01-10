import argparse
import os
import json
import numpy as np


def save_avg_knn_acc(acc, output_dir):
    out_path = os.path.join(output_dir, "avg_top1_knn_accuracy.json")
    with open(out_path, 'w') as file:
        json.dump({"top1_knn_accuracy": acc}, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", "-e", type=str, help="Path where the metrics per fold are stored",
                        default="/home/bdolicki/thesis/hissl-logs/linear_bach_dino/8532526")
    args = parser.parse_args()

    num_folds = len([folder for folder in os.listdir(args.experiment_path) if "fold" in folder])
    print("Number of folds", num_folds)
    fold_accs = []
    for fold in range(num_folds):
        acc_path = os.path.join(args.experiment_path, f"fold{fold}/top1_knn_accuracy.json")
        with open(acc_path) as json_file:
            fold_acc = json.load(json_file)
        fold_accs.append(fold_acc["top1_knn_accuracy"])
    avg_acc = np.mean(fold_accs)
    print("Average knn accuracy", avg_acc)

    out_path = os.path.join(args.experiment_path, "avg")
    os.makedirs(out_path, exist_ok=True)

    save_avg_knn_acc(avg_acc, out_path)
