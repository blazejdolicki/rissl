import json
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, help="Directory with logs: checkpoints, parameters, metrics")
args = parser.parse_args()

jsonl_path = os.path.join(args.log_dir, "stdout.json")

with open(jsonl_path, 'r') as json_file:
    json_list = list(json_file)

num_epochs = json.loads(json_list[-1])["ep"] + 1
epoch_losses = np.zeros(num_epochs)
for json_str in json_list:
    iteration_logs = json.loads(json_str)
    epoch = int(iteration_logs["ep"])
    loss = iteration_logs["loss"]
    epoch_losses[epoch] = loss

losses = {"best_epoch": int(np.argmin(epoch_losses)),
          "best_loss": float(np.min(epoch_losses)),
          "epoch_losses": list(epoch_losses)}

print("losses", losses)

losses_path = os.path.join(args.log_dir, "losses_summary.json")
with open(losses_path, 'w') as file:
    json.dump(losses, file, indent=4)