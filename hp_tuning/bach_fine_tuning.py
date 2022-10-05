import subprocess
import argparse
import os
from hp_utils import sample_runs, save_runs

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="logs",
                    help="Directory with logs: checkpoints, parameters, metrics")
parser.add_argument('--exp_name', type=str, default="hp_tuning",
                    help='MLFlow experiment folder where the results will be logged')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--no_rotation_transforms', action="store_true",
                    help="If this argument is specified, don't use rotations as image transformations.")
parser.add_argument("--model_type", type=str)
parser.add_argument('--old_img_path_prefix', type=str,
                        help='Old path to images that will be replaced by the new prefix in the .npy files.'
                             'It is specifically useful when the .npy files where generated for one directory'
                             'and now they should be used in another.')
parser.add_argument('--new_img_path_prefix', type=str,
                        help='New path to images that will be replaced by the new prefix.in the .npy files.'
                             'It is specifically useful when the .npy files where generated for one directory'
                             'and now they should be used in another.')
parser.add_argument("--weight_decay", type=str)
parser.add_argument("--checkpoint_path", type=str, help="Checkpoint of a pretrained model")

args = parser.parse_args()

job_id = args.log_dir.split("/")[-1]
# random seed for sampling runs

search = {"max_lrs": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]}
hyperparams = [search[hp] for hp in search]

num_runs = 6
runs = sample_runs(num_runs, *hyperparams)

import numpy as np
runs = np.array(runs)

save_runs(search, args.log_dir, num_runs, runs)

k = 5
for i, run in enumerate(runs):
    log_dir = os.path.join(args.log_dir, f"run_{i}")
    for fold in range(k):
        fold_dir = os.path.join(log_dir, f"fold{fold}")
        subprocess.run(["python", "train.py",
                        "--dataset", "bach",
                        "--data_dir", args.data_dir,
                        "--old_img_path_prefix", args.old_img_path_prefix,
                        "--new_img_path_prefix", args.new_img_path_prefix,
                        "--log_dir", fold_dir,
                        "--job_id", job_id,
                        "--exp_name", args.exp_name,
                        "--mlflow_dir", "/home/b.dolicki/mlflow_runs",
                        "--fold", str(fold),
                        "--model_type", args.model_type,
                        "--batch_size", str(64),
                        "--num_workers", str(1),
                        "--num_epochs", str(100),
                        "--max_lr", str(run[0]),
                        "--weight_decay", args.weight_decay,
                        "--lr_scheduler_type", "Constant",
                        "--optimizer", "adam",
                        "--fixparams",
                        "--checkpoint_path", args.checkpoint_path
                        ]+(["--no_rotation_transforms"] if args.no_rotation_transforms else []))
    subprocess.run(["python", "tools/aggregate_fold_results_rissl.py",
                    "--log_dir", str(log_dir),
                    "--exp_name", args.exp_name,
                    "--mlflow_dir", "/home/b.dolicki/mlflow_runs",
                    ])




