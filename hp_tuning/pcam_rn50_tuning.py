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
          "lr_scheduler_types": ["StepLR", "OneCycleLR"],
          "optimizers": ["adam", "sgd"]}
hyperparams = [search[hp] for hp in search]

num_runs = 9 # TODO change to 24
# runs = sample_runs(num_runs, *hyperparams)
runs = [['0.001', '0.0001', 'StepLR', 'adam'],
       ['0.0001', '1e-06', 'StepLR', 'adam'],
       ['0.001', '1e-05', 'OneCycleLR', 'adam'],
       ['0.001', '1e-05', 'StepLR', 'sgd'],
       ['0.1', '0.0001', 'OneCycleLR', 'adam'],
       ['0.1', '0.0001', 'StepLR', 'sgd'],
       ['0.001', '1e-05', 'OneCycleLR', 'sgd'],
       ['0.0001', '1e-06', 'OneCycleLR', 'adam'],
       ['0.1', '1e-06', 'StepLR', 'sgd']]

import numpy as np
runs = np.array(runs)

save_runs(search, args.log_dir, num_runs, runs)
i = 15
for _, run in enumerate(runs):
    log_dir = os.path.join(args.log_dir, f"run_{i}")
    subprocess.run(["python", "train.py",
                    "--dataset", "pcam",
                    "--data_dir", args.data_dir,
                    "--log_dir", log_dir,
                    "--job_id", job_id,
                    "--exp_name", args.exp_name,
                    "--mlflow_dir", "/home/b.dolicki/mlflow_runs",
                    "--model_type", "resnext50_32x4d",
                    "--batch_size", str(512),
                    "--num_workers", str(1),
                    "--num_epochs", str(50), # TODO Change to 50
                    "--max_lr", str(run[0]),
                    "--weight_decay", str(run[1]),
                    "--lr_scheduler_type", str(run[2]),
                    "--optimizer", str(run[3])
                    ])
    i += 1
