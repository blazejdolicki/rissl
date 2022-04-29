import subprocess
import argparse
import os
from hp_utils import sample_runs

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="logs",
                    help="Directory with logs: checkpoints, parameters, metrics")
parser.add_argument('--exp_name', type=str, default="hp_tuning",
                    help='MLFlow experiment folder where the results will be logged')
args = parser.parse_args()

max_lrs = [0.1, 0.01, 0.001, 0.0001]
weight_decays = [0.0001, 0.00001, 0.000001]
lr_scheduler_types = ["StepLR", "OneCycleLR"]
optimizers = ["adam", "sgd"]

num_runs = 5 # TODO: Change after testing
hyperparams = [max_lrs, weight_decays, lr_scheduler_types, optimizers]
runs = sample_runs(num_runs, *hyperparams)

for i, run in enumerate(runs):
    log_dir = os.path.join(args.log_dir, f"run_{i}")
    subprocess.run(["python", "train.py",
                    "--dataset", "pcam",
                    "--log_dir", str(log_dir),
                    "--exp_name", str(args.exp_name),
                    "--model_type", "resnext50_32x4d",
                    "--num_epochs", str(1), # TODO change to 1 after testing
                    "--max_lr", str(run[0]),
                    "--wd", str(run[1]),
                    "--lr_scheduler_type", str(runs[2]),
                    "--optimizer", str(runs[3])
                    ])
