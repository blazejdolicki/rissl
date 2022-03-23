import subprocess
import argparse
import os
import json
import numpy as np
from argparse import Namespace
import sys
import pandas as pd
import torch

# hack to import evaluate function - works only on Lisa cluster
sys.path.append('/home/bdolicki/thesis/ssl-histo')
from evaluate import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory with logs: checkpoints, parameters, metrics")
    parser.add_argument('--exp_name', type=str, default="sup_breakhis_scale",
                        help='MLFlow experiment folder where the results will be logged')
    parser.add_argument('--best_or_last', type=str, default="best", choices=["best", "last"],
                        help="Evaluate on the model from last epoch or from the epoch with the best validation score")
    parser.add_argument('--job_id', type=str, help="SLURM job id")
    args = parser.parse_args()

    # best found hyperparameters
    max_lr = 0.001
    model_type = "resnext101_32x8d"
    num_epochs = 50

    # magnitudes in Breakhis dataset
    mags = [40, 100, 200, 400]
    folds = [1, 2, 3, 4, 5]
    # initialize matrix with results over folds
    results = np.zeros((len(folds), len(mags), len(mags)))

    for fold_idx, fold in enumerate(folds):
        for train_mag_idx, train_mag in enumerate(mags):
            train_log_dir = os.path.join(args.log_dir, f"fold_{fold}", f"train_mag_{train_mag}")
            subprocess.run(["python", "train.py",
                            "--dataset", "breakhis_fold",
                            "--log_dir", str(train_log_dir),
                            "--exp_name", str(args.exp_name),
                            "--max_lr", str(max_lr),
                            "--model_type", str(model_type),
                            "--num_epochs", str(num_epochs),
                            "--fold", str(fold),
                            "--job_id", str(args.job_id),
                            "--train_mag", str(train_mag),
                            "--test_mag", str(train_mag) # evaluate on the same magnitude
                            ])
            # TODO: should it be best model or final model?
            model_name = "best_model.pt" if args.best_or_last == "best" else "final_model.pt"
            model_path = os.path.join(train_log_dir, "checkpoints", model_name)
            for test_mag_idx, test_mag in enumerate(mags):
                test_log_dir = os.path.join(train_log_dir, f"test_mag_{test_mag}")
                eval_args = {"log_dir": test_log_dir,
                             "exp_name": args.exp_name,
                             "checkpoint_path": model_path,
                             "split": "test",
                             "test_mag": test_mag,
                             "fold": fold}
                eval_args = Namespace(**eval_args)
                loss, acc = evaluate(eval_args)
                results[fold_idx][train_mag_idx][test_mag_idx] = acc

    # save results per fold
    results_path = os.path.join(args.log_dir, "results.pt")
    torch.save(results, results_path)

    # average over folds
    avg_results = np.mean(results, axis=0)
    avg_results = pd.DataFrame(avg_results, columns=mags, index=mags)
    print("Average results over folds")
    print(avg_results)
    avg_results_path = os.path.join(args.log_dir, "avg_results")
    with open(f"{avg_results_path}_latex.txt", "w") as f:
        f.write(avg_results.to_latex())
    avg_results.to_csv(f"{avg_results_path}.csv")

    std_results = np.std(results, axis=0)
    std_results = pd.DataFrame(std_results, columns=mags, index=mags)
    print("Standard deviation of results over folds")
    print(std_results)
    std_results_path = os.path.join(args.log_dir, "std_results")
    with open(f"{std_results_path}_latex.txt", "w") as f:
        f.write(std_results.to_latex())
    std_results.to_csv(f"{std_results_path}.csv")
