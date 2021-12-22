import yaml
import pprint
import argparse
from collections import defaultdict


def list2str(param):
    """
    Some parameters from .yaml files are read as lists and incorrectly saved, this function fixes the formatting.
    :param param:
    :return:
    """
    return str(param).replace("'", "")

def convert_config(train_config, default_linear_config):
    """
    Create a generic configuration for linear evaluation based on a custom pretraining config and a default linear eval
    config. Run-specific parameters related to the dataset or model weights should be ovewritten in the job script.
    :param train_config: custom pretraining config
    :param default_linear_config: default linear eval
    config
    :return:generic configuration for linear evaluation
    """
    # initialize dict

    linear_config = train_config

    # remove arguments from train config if they exists
    try:
        del linear_config["config"]["DATA"]["TRAIN"]["COLLATE_FUNCTION"]
        del linear_config["config"]["MODEL"]["TEMP_FROZEN_PARAMS_ITER_MAP"]
    except KeyError:
        pass

    # use validation set
    linear_config["config"]["TEST_MODEL"] = True

    # use only RandomResizedCrop and RandomHorizontalFlip for transformations for linear classification
    linear_config["config"]["DATA"]["TRAIN"]["TRANSFORMS"] = default_linear_config["config"]["DATA"]["TRAIN"]["TRANSFORMS"]
    linear_config["config"]["DATA"]["TEST"] = {} # init test dictionary
    linear_config["config"]["DATA"]["TEST"]["TRANSFORMS"] = default_linear_config["config"]["DATA"]["TEST"]["TRANSFORMS"]

    # use the same placeholder values for test set, it's overwritten in the script anyway

    linear_config["config"]["DATA"]["TEST"]["DATA_SOURCES"] = default_linear_config["config"]["DATA"]["TRAIN"]["DATA_SOURCES"]
    linear_config["config"]["DATA"]["TEST"]["DATASET_NAMES"] = default_linear_config["config"]["DATA"]["TRAIN"]["DATASET_NAMES"]
    linear_config["config"]["DATA"]["TEST"]["LABEL_TYPE"] = train_config["config"]["DATA"]["TRAIN"]["LABEL_TYPE"]
    # use the same batch size as for the training set and for the pretraining
    # TODO: maybe increase the batch size later
    linear_config["config"]["DATA"]["TEST"]["BATCHSIZE_PER_REPLICA"] = linear_config["config"]["DATA"]["TRAIN"]["BATCHSIZE_PER_REPLICA"]


    # set up the evaluation settings
    linear_config["config"]["MODEL"]["FEATURE_EVAL_SETTINGS"] = {
        "EVAL_MODE_ON": True,
        "FREEZE_TRUNK_ONLY": True,
    }

    # set up the linear head
    assert "VISION_TRANSFORMERS" in linear_config["config"]["MODEL"]["TRUNK"], \
        "This script was made for transformer backbones, do not use it out of the box for CNNs"

    hidden_dim = linear_config["config"]["MODEL"]["TRUNK"]["VISION_TRANSFORMERS"]["HIDDEN_DIM"]
    # TODO change number of classes to set based on the dataset instead of hardcoding
    num_classes = 9 # hardcoded for NCT

    linear_config["config"]["MODEL"]["HEAD"]["PARAMS"] = [
        ["mlp", {"dims": [hidden_dim, num_classes]}],
      ]

    # set standard cross-entropy loss for classification
    linear_config["config"]["LOSS"] = default_linear_config["config"]["LOSS"]

    # set the optimizer as the default for now
    # TODO make sure your really understand the learning rate schedule and tune it for non-ImageNet datasets
    linear_config["config"]["OPTIMIZER"] = default_linear_config["config"]["OPTIMIZER"]

    # set the accuracy meters
    linear_config["config"]["METERS"] = default_linear_config["config"]["METERS"]

    # set high checkpoint freq to never save model checkpoints for linear evaluation
    linear_config["config"]["CHECKPOINT"]["CHECKPOINT_FREQUENCY"] = 1000
    linear_config["config"]["HOOKS"]["TENSORBOARD_SETUP"]["LOG_PARAMS"] = False
    return linear_config

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_path", type=str, help="Path to corresponding configuration of training",
                        required=False, default='config/blazej/pretrain/train_dino.yaml')
    parser.add_argument("--default_linear_config_path", type=str, help="Path to default linear evaluation path",
                        default="config/blazej/benchmark/linear/deit_s16_imagenet.yaml")
    args = parser.parse_args()

    train_config_name = args.train_config_path.split("/")[-1]
    linear_config_name = train_config_name.replace("train", "linear")
    linear_config_dir = args.train_config_path.replace("pretrain", "benchmark/linear").replace(train_config_name, "")

    os.makedirs(linear_config_dir, exist_ok=True)
    linear_config_name = os.path.join(linear_config_dir, linear_config_name)

    # load training yaml file
    with open(args.train_config_path) as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    # load default linear yaml file
    with open(args.default_linear_config_path) as f:
        default_linear_config = yaml.load(f, Loader=yaml.FullLoader)

    linear_config = convert_config(train_config, default_linear_config)

    print(f"Converting training config `{args.train_config_path}`")
    print(f"into linear evaluation config `{linear_config_name}`")

    # save linear yaml file
    with open(linear_config_name, 'w') as file:
        yaml.dump(linear_config, file)

