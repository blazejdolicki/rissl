import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="logs",
                    help="Directory with logs: checkpoints, parameters, metrics")
parser.add_argument('--exp_name', type=str, default="hp_tuning",
                    help='MLFlow experiment folder where the results will be logged')
args = parser.parse_args()

# best found hyperparameters
max_lr = 0.001
model_type = "resnext101_32x8d"
num_epochs = 50

mags = [40, 100, 200, 400]
for train_mag in mags:
    for test_mag in mags:
        log_dir = os.path.join(args.log_dir, f"train_mag_{train_mag}_test_mag_{test_mag}")
        subprocess.run(["python", "train.py",
                        "--dataset", "breakhis_fold",
                        "--log_dir", str(log_dir),
                        "--exp_name", str(args.exp_name),
                        "--max_lr", str(max_lr),
                        "--model_type", str(model_type),
                        "--num_epochs", str(num_epochs),
                        "--train_mag", str(train_mag),
                        "--test_mag", str(test_mag)
                        ])
