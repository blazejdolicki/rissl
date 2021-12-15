import yaml
import pprint
import argparse
from collections import defaultdict


def convert_config(train_config, weights_path):
    # initialize dict
    extract_config = {"config": {
        "DATA": {},
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

    # indicate feature extraction
    extract_config["engine_name"] = "extract_features"
    # use the same data, trunk, distributed and machine settings as the trained model
    extract_config["config"]["DATA"] = train_config["config"]["DATA"]
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

    # add saved weights from trained model
    extract_config["config"]["MODEL"]["WEIGHTS_INIT"]["PARAM_FILE"] = weights_path
    return extract_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, help="Path to weights of the trained VISSL model in .torch format")
    parser.add_argument("--train_config_path", type=str, help="Path to corresponding configuration of training",
                        required=False, default='configs/config/dummy/blazej_quick_cpu_resnet50_simclr_on_dummy.yaml')
    parser.add_argument("--extract_config_path", type=str,
                        help="Path where the feature extraction config file will be output",
                        required=False, default="configs/config/dummy/feat_extract.yaml")
    args = parser.parse_args()

    # load training yaml file
    with open(args.train_config_path) as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
        pprint.pprint(train_config)

    extract_config = convert_config(train_config, args.weights_path)

    # save extraction yaml file
    with open(args.extract_config_path, 'w') as file:
        yaml.dump(extract_config, file)
