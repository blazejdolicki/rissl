import yaml

def list2str(param):
    """
    Some parameters from .yaml files are read as lists and incorrectly saved, this function fixes the formatting.
    :param param:
    :return:
    """
    return str(param).replace("'", "")

def convert_config(train_config, default_finetune_config):
    """
    Create a generic configuration for finetuning evaluation based on a custom pretraining config and a default finetuning
    config. Run-specific parameters related to the dataset or model weights should be ovewritten in the job script.
    :param train_config: custom pretraining config
    :param default_finetune_config: default finetune eval
    config
    :return:generic configuration for finetune evaluation
    """

    finetune_config = train_config

    # remove pretraining-specific arguments from train config if they exists
    try:
        del finetune_config["config"]["DATA"]["TRAIN"]["COLLATE_FUNCTION"]
    except KeyError:
        pass

    # use validation set
    finetune_config["config"]["TEST_MODEL"] = True

    # use only RandomResizedCrop and RandomHorizontalFlip for transformations for finetune classification
    finetune_config["config"]["DATA"]["TRAIN"]["TRANSFORMS"] = default_finetune_config["config"]["DATA"]["TRAIN"][
        "TRANSFORMS"]
    finetune_config["config"]["DATA"]["TEST"] = {}  # init test dictionary
    finetune_config["config"]["DATA"]["TEST"]["TRANSFORMS"] = default_finetune_config["config"]["DATA"]["TEST"][
        "TRANSFORMS"]

    # use the same placeholder values for test set, it's overwritten in the script anyway
    finetune_config["config"]["DATA"]["TEST"]["DATA_SOURCES"] = default_finetune_config["config"]["DATA"]["TRAIN"][
        "DATA_SOURCES"]
    finetune_config["config"]["DATA"]["TEST"]["DATASET_NAMES"] = default_finetune_config["config"]["DATA"]["TRAIN"][
        "DATASET_NAMES"]
    finetune_config["config"]["DATA"]["TEST"]["LABEL_TYPE"] = train_config["config"]["DATA"]["TRAIN"]["LABEL_TYPE"]
    # use the same batch size as for the training set and for the pretraining
    # TODO: maybe increase the batch size later
    finetune_config["config"]["DATA"]["TEST"]["BATCHSIZE_PER_REPLICA"] = finetune_config["config"]["DATA"]["TRAIN"][
        "BATCHSIZE_PER_REPLICA"]

    # set up the evaluation settings
    finetune_config["config"]["MODEL"]["FEATURE_EVAL_SETTINGS"] = {
        "EVAL_MODE_ON": True,
        "EVAL_TRUNK_AND_HEAD": False,
    }

    # TODO change number of classes to set based on the dataset instead of hardcoding
    num_classes = 2  # hardcoded for PCAM
    hidden_dim = 512 # hardcoded for resnet18

    finetune_config["config"]["MODEL"]["HEAD"]["PARAMS"] = [
        ["mlp", {"dims": [hidden_dim, num_classes]}],
    ]

    # set standard cross-entropy loss for classification
    finetune_config["config"]["LOSS"] = default_finetune_config["config"]["LOSS"]

    # for now: use the default optimizer and learning rate schedule
    # TODO think more about the learning rate schedule, maybe use the same as for supervised training
    finetune_config["config"]["OPTIMIZER"] = default_finetune_config["config"]["OPTIMIZER"]

    # set the accuracy meters
    finetune_config["config"]["METERS"] = default_finetune_config["config"]["METERS"]

    # set high checkpoint freq to never save model checkpoints for finetune evaluation
    # finetune_config["config"]["CHECKPOINT"]["CHECKPOINT_FREQUENCY"] = 1000
    # don't log model parameters
    finetune_config["config"]["HOOKS"]["TENSORBOARD_SETUP"]["LOG_PARAMS"] = False
    return finetune_config

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_path", type=str, help="Path to corresponding configuration of pretraining",
                        required=False, default='config/pretrain/simclr/simclr_resnet.yaml')
    parser.add_argument("--default_finetune_config_path", type=str, help="Path to default finetuning path",
                        default="config/benchmark/fulltune/imagenet1k/eval_resnet_8gpu_transfer_in1k_fulltune.yaml")
    args = parser.parse_args()

    train_config_name = args.train_config_path.split("/")[-1]
    finetune_config_name = train_config_name.replace("train", "finetune")
    finetune_config_dir = args.train_config_path.replace("pretrain", "benchmark/finetune").replace(train_config_name, "")

    os.makedirs(finetune_config_dir, exist_ok=True)
    finetune_config_name = os.path.join(finetune_config_dir, finetune_config_name)

    # load training yaml file
    with open(args.train_config_path) as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    # load default finetune yaml file
    with open(args.default_finetune_config_path) as f:
        default_finetune_config = yaml.load(f, Loader=yaml.FullLoader)

    finetune_config = convert_config(train_config, default_finetune_config)

    print(f"Converting training config `{args.train_config_path}`")
    print(f"into finetune evaluation config `{finetune_config_name}`")

    # save finetune yaml file
    with open(finetune_config_name, 'w') as file:
        yaml.dump(finetune_config, file)