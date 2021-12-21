import yaml
import pprint
import argparse
from collections import defaultdict


def convert_config(train_config):
    # initialize dict
    extract_config = {"config": {
        "MODEL": {"TRUNK": {},
                  "FEATURE_EVAL_SETTINGS": {},
                  "WEIGHTS_INIT": {
                      "PARAM_FILE": {}
                  }},
        "DISTRIBUTED": {},
        "MACHINE": {},
        "EXTRACT_FEATURES": {"OUTPUT_DIR": "",
                             "CHUNK_THRESHOLD": 0}
    }}

    # remove arguments from train config if they exists
    try:
        del train_config["config"]["DATA"]["TRAIN"]["COLLATE_FUNCTION"]
    except KeyError:
        pass

    # indicate feature extraction
    extract_config["engine_name"] = "extract_features"
    # use the same data, trunk, distributed and machine settings as the trained model
    extract_config["config"]["DATA"] = train_config["config"]["DATA"]
    # add test
    extract_config["config"]["DATA"]["TEST"] = {}
    # set the same batch size for test and train
    extract_config["config"]["DATA"]["TEST"]["BATCHSIZE_PER_REPLICA"] = train_config["config"]["DATA"]["TRAIN"]["BATCHSIZE_PER_REPLICA"]

    for split in ["TRAIN", "TEST"]:
        # use standard feature extraction transformations, source:
        # https://github.com/facebookresearch/vissl/blob/main/configs/config/feature_extraction/extract_resnet_in1k_8gpu.yaml
        extract_config["config"]["DATA"][split]["TRANSFORMS"] = [
            {"name": "Resize",
             "size": 256},
            {"name": "CenterCrop",
             "size": 224},
            {"name": "ToTensor"},
            {"name": "Normalize",
             "mean": [0.485, 0.456, 0.406],
             "std": [0.229, 0.224, 0.225]}
        ]

    extract_config["config"]["MODEL"]["TRUNK"] = train_config["config"]["MODEL"]["TRUNK"]
    extract_config["config"]["DISTRIBUTED"] = train_config["config"]["DISTRIBUTED"]
    extract_config["config"]["MACHINE"] = train_config["config"]["MACHINE"]

    # set up the extraction settings
    extract_config["config"]["MODEL"]["FEATURE_EVAL_SETTINGS"] = {
        "EVAL_MODE_ON": True,
        "FREEZE_TRUNK_ONLY": True,
        "EXTRACT_TRUNK_FEATURES_ONLY": True,
        "SHOULD_FLATTEN_FEATS": False
    }

    return extract_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_path", type=str, help="Path to corresponding configuration of training",
                        required=False, default='configs/config/dummy/blazej_quick_cpu_resnet50_simclr_on_dummy.yaml')
    parser.add_argument("--extract_config_path", type=str,
                        help="Path where the feature extraction config file will be output",
                        required=False, default="configs/config/dummy/feat_extract.yaml")
    args = parser.parse_args()

    # load training yaml file
    with open(args.train_config_path) as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
        # pprint.pprint(train_config)

    feat_extract_config = convert_config(train_config)

    print(f"Converting training config `{args.train_config_path}` into feature extraction config `{args.extract_config_path}`")

    # save extraction yaml file
    with open(args.extract_config_path, 'w') as file:
        yaml.dump(feat_extract_config, file)
