import argparse
from argparse import Namespace
import os
import json
import logging
import argparse
import logging
import sys

from convert_vissl_to_torchvision import convert_and_save_model

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, help="Directory with logs including the best model checkpoint")
parser.add_argument("--num_epochs", type=int, help="Number of epochs for which the model was trained")
args = parser.parse_args()

checkpoint_name = f"model_final_checkpoint_phase{args.num_epochs-1}.torch"
args.model_url_or_file = os.path.join(args.log_dir, checkpoint_name)
args.output_dir = args.log_dir
args.output_name = "final_model"
args.include_head = True

logging.info(f"Converting final model from {args.model_url_or_file}")
convert_and_save_model(args, replace_prefix="_feature_blocks.")
