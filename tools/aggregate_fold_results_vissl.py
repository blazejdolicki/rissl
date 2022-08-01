import json
import numpy as np
import pprint
import matplotlib.pyplot as plt
import os
import argparse


def save_avg_acc(avg_accs_over_epochs, exp_path):
    out_path = os.path.join(exp_path, "avg_accs_over_epochs.json")
    with open(out_path, 'w') as file:
        json.dump(avg_accs_over_epochs, file, indent=4)


def plot_avg_acc(num_epochs, avg_accs_over_epochs, exp_path):
    for split in avg_accs_over_epochs:
        plt.plot(list(range(num_epochs)), avg_accs_over_epochs[split], '-', label=split)

    plt.title("Average fold train and test accuracy over epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    out_path = os.path.join(exp_path, "plot_avg_accs_over_epochs.jpg")
    plt.savefig(out_path, dpi=200)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", "-e", type=str, help="Path where the metrics per fold are stored",
                        default="/home/bdolicki/thesis/hissl-logs/linear_bach_dino/8532526")
    args = parser.parse_args()

    num_folds = len([folder for folder in os.listdir(args.experiment_path) if "fold" in folder])
    print("Number of folds", num_folds)
    fold_metrics = {}
    for fold in range(num_folds):
        metrics_path = os.path.join(args.experiment_path, f"fold{fold}/metrics.json")
        fold_metrics[fold] = {"train": {},
                              "test": {}}
        with open(metrics_path) as json_file:
            fold_phase_metrics = list(json_file)
            for train_phase_idx in range(0, len(fold_phase_metrics), 2):
                epoch = int(train_phase_idx / 2)
                test_phase_idx = train_phase_idx + 1

                fold_metrics[fold]["train"][epoch] = json.loads(fold_phase_metrics[train_phase_idx])
                fold_metrics[fold]["test"][epoch] = json.loads(fold_phase_metrics[test_phase_idx])
    num_epochs = len(fold_metrics[0]["train"])
    print("Number of epochs:", num_epochs)
    avg_accs_over_epochs = {"train": [],
                            "test": []}

    # average metrics
    for split in ["train", "test"]:
        for epoch in range(num_epochs):
            epoch_accs_over_folds = [fold_metrics[fold][split][epoch][f"{split}_accuracy_list_meter"]["top_1"]["0"]
                                     for fold in range(num_folds)]

            avg_accs_over_epochs[split].append(np.mean(epoch_accs_over_folds))
    print("Average accuracies over epochs")
    print(pprint.pprint(avg_accs_over_epochs))

    # save results as dict and plot
    out_path = os.path.join(args.experiment_path, "avg")
    os.makedirs(out_path, exist_ok=True)

    save_avg_acc(avg_accs_over_epochs, out_path)
    plot_avg_acc(num_epochs, avg_accs_over_epochs, out_path)