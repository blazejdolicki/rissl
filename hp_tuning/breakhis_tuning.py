import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="logs",
                    help="Directory with logs: checkpoints, parameters, metrics")
parser.add_argument('--exp_name', type=str, default="hp_tuning",
                    help='MLFlow experiment folder where the results will be logged')
args = parser.parse_args()

max_lrs = [0.1, 0.01, 0.001]
model_types = ["resnet18",
               "resnet34",
               "resnet50",
               "resnet101",
               "resnet152",
               "resnext50_32x4d",
               "resnext101_32x8d",
               "wide_resnet50_2",
               "wide_resnet101_2"]

for max_lr in max_lrs:
    for model_type in model_types:
        log_dir = os.path.join(args.log_dir, f"lr_{max_lr}_model_{model_type}")
        subprocess.run(["python", "train.py",
                        "--dataset", "breakhis",
                        "--log_dir", str(log_dir),
                        "--exp_name", str(args.exp_name),
                        "--max_lr", str(max_lr),
                        "--model_type", str(model_type),
                        "--num_epochs", str(50),
                        "--patience", str(20)
                        ])
