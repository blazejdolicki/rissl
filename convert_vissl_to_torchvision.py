# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Use this script to convert VISSL ResNe(X)ts models to match Torchvision exactly.
Copied from: https://github.com/facebookresearch/vissl/blob/main/extra_scripts/convert_vissl_to_torchvision.py
"""

import argparse
import logging
import os
import sys
import yaml
import torch
import re
from iopath.common.file_io import g_pathmgr
from pathlib import Path
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from vissl.utils.checkpoint import replace_module_prefix
from vissl.utils.io import is_url

from models import get_model


# initiate the logger
FORMAT = "%(asctime)s [%(levelname)s - %(filename)s: %(lineno)4d] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def convert2torch(epoch_model, args):

    # load the model
    model_path = os.path.join(args.vissl_path, epoch_model)
    if is_url(model_path):
        logger.info(f"Loading from url: {model_path}")
        model = load_state_dict_from_url(model_path)
    else:
        model = torch.load(model_path, map_location=torch.device("cpu"))

    # get the model trunk to rename
    if "classy_state_dict" in model.keys():
        model_trunk = model["classy_state_dict"]["base_model"]["model"]["trunk"]
    elif "model_state_dict" in model.keys():
        model_trunk = model["model_state_dict"]
    else:
        model_trunk = model
    logger.info(f"Input model loaded. Number of params: {len(model_trunk.keys())}")

    if args.include_head:
        model_head = model["classy_state_dict"]["base_model"]["model"]["heads"]
        model_head = replace_module_prefix(model_head, prefix="0.clf.0.", replace_with="fc.")
        model_trunk.update(model_head)

    # convert the trunk
    converted_model = replace_module_prefix(model_trunk, "_feature_blocks.")

    logger.info(f"Converted model to torch. Number of params: {len(converted_model.keys())}")

    output_model_filepath = os.path.join(args.torch_path, epoch_model)
    logger.info(f"Saving model: {output_model_filepath}")
    torch.save(converted_model, output_model_filepath)
    logger.info("DONE!")

    return converted_model


def convert2jit(epoch_model, torch_state_dict, args):
    args = get_model_args_from_pretrained_checkpoint(args)
    args.headless = True
    print("pretrained args")
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # select a model, num_classes=None because we're using extracting features
    model = get_model(model_type=args.model_type, num_classes=None, args=args).to(device)
    model.load_state_dict(torch_state_dict)

    # prepare model with TorchScript
    batch_size = 4
    img_width = img_height = 16
    img_channels = 3
    example = torch.rand(batch_size, img_channels, img_height, img_width)
    model = model.to("cpu")
    traced_script_module = torch.jit.trace(model, example)

    # save the TorchScript model
    ts_model_path = os.path.join(args.jit_path, epoch_model)
    traced_script_module.save(ts_model_path)

    logger.info(f"Saved model: {ts_model_path}")


def get_model_args_from_pretrained_checkpoint(args):
    with open(os.path.join(args.checkpoint_path, "train_config.yaml")) as yaml_file:
        train_config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    model_family = train_config["MODEL"]["TRUNK"]["NAME"]
    model_args = train_config["MODEL"]["TRUNK"][(model_family + "s").upper()]
    args.model_type = model_family + str(model_args["DEPTH"])

    if is_equivariant(args.model_type):
        for name, model_arg in model_args.items():
            args.__setattr__(name, model_arg)

    return args


def is_equivariant(model_type):
    return "e2" in model_type


def main():
    parser = argparse.ArgumentParser(
        description="Convert VISSL ResNe(X)ts models to Torchvision"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        required=True,
        help="Model directory that contains the state dicts",
    )

    parser.add_argument(
        "--include_head", action="store_true",
        help="If specified, includes head in the state dict. Useful when converting the model after finetuning "
             "or training a linear classifier on top of the pretrained model."
    )

    parser.add_argument("--every_n", type=int, default=20, help="Save model after every n epochs")

    args = parser.parse_args()

    assert os.path.isdir(args.checkpoint_path)

    args.vissl_path = os.path.join(args.checkpoint_path, "vissl")
    args.torch_path = os.path.join(args.checkpoint_path, "torch")
    args.jit_path = os.path.join(args.checkpoint_path, "jit")

    # create new directories
    os.makedirs(args.torch_path)
    os.makedirs(args.jit_path)

    for epoch_model in os.listdir(args.vissl_path):
        try:
            # extract epoch, add one because index starts at 0
            epoch = int(re.findall(r'\d+', epoch_model)[0]) + 1
            if epoch % args.every_n == 0 or "final" in epoch_model:
                torch_state_dict = convert2torch(epoch_model, args)
                convert2jit(epoch_model, torch_state_dict, args)
        except IndexError as e: # there is always some random "checkpoint.torch" file in the directory
            print(f"IndexError: {e}. Epoch number not found for checkpoint file: \"{epoch_model}\"")


if __name__ == "__main__":
    main()
