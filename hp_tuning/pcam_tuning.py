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

args = parser.parse_args()

job_id = args.log_dir.split("/")[-1]
# random seed for sampling runs


search = {"max_lrs": [0.1, 0.01, 0.001, 0.0001],
          "weight_decays": [0.0001, 0.00001, 0.000001],
          "optimizers": ["adam", "sgd"]}
hyperparams = [search[hp] for hp in search]

num_runs = 24
runs = sample_runs(num_runs, *hyperparams)

import numpy as np
runs = np.array(runs)

save_runs(search, args.log_dir, num_runs, runs)

for i, run in enumerate(runs):
    log_dir = os.path.join(args.log_dir, f"run_{i}")
    subprocess.run(["python", "train.py",
                    "--dataset", "pcam",
                    "--data_dir", args.data_dir,
                    "--log_dir", log_dir,
                    "--job_id", job_id,
                    "--exp_name", args.exp_name,
                    "--mlflow_dir", "/home/b.dolicki/mlflow_runs",
                    "--model_type", "resnet18",
                    "--batch_size", str(512),
                    "--num_workers", str(1),
                    "--num_epochs", str(50), # TODO Change to 50
                    "--max_lr", str(run[0]),
                    "--weight_decay", str(run[1]),
                    "--lr_scheduler_type", "StepLR",
                    "--optimizer", str(run[3])
                    ])
