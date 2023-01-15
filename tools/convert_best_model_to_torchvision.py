import os
import json
import argparse
import logging
import sys
sys.path.insert(0, "/projects/rissl/blazej/thesis/")
from rissl.convert_vissl_to_torchvision import convert_and_save_model

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, help="Directory with logs including the best model checkpoint")

args = parser.parse_args()

with open(os.path.join(args.log_dir, "acc_summary.json")) as json_file:
    acc_summary = json.load(json_file)



def is_final_best(acc_summary):
    return acc_summary["best_valid_acc"] == acc_summary["final_valid_acc"]


checkpoint_name = f"model_{('final_checkpoint_' if is_final_best(acc_summary) else '')}phase{acc_summary['best_valid_epoch']}.torch"
args.model_url_or_file = os.path.join(args.log_dir, checkpoint_name)
args.output_dir = args.log_dir
args.output_name = "best_model"
args.include_head = True

logging.info(f"Converting best model from {args.model_url_or_file}")
convert_and_save_model(args, replace_prefix="_feature_blocks.")
