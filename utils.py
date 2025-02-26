import logging
import os
import sys
import torch
import numpy as np
import random
import mlflow
import copy
import torchvision.transforms.functional as F
import yaml
from argparse import Namespace
from pathlib import Path
import json
from torchvision import transforms
from torchmetrics import AUROC, ConfusionMatrix, F1Score

from models import models, is_equivariant
from collect_env import collect_env_info
from datasets import convert_transform_to_dict, convert_dict_to_transform


def setup_logging(output_dir):
    """
    Setup various logging streams: stdout and file handlers.
    Suitable for single GPU only.
    """
    # get the filename if we want to log to the file as well
    log_filename = f"{output_dir}/log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.getLogger('PIL').setLevel(logging.WARNING)


def setup_mlflow(args):
    job_id = args.log_dir.split("/")[-1]
    mlflow.set_tracking_uri(f"file:///{args.mlflow_dir}")
    mlflow.set_experiment(args.exp_name)
    mlflow.start_run(run_name=job_id)
    mlflow_args = copy.deepcopy(args)
    mlflow_args.transform = "See args.json"
    mlflow.log_params(vars(mlflow_args))


def save_env_info(logdir):
    env_info_path = os.path.join(logdir, "env_info.txt")
    with open(env_info_path, 'w') as f:
        f.write(collect_env_info())


def fix_seed(seed):
    logging.info(f"Random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args(parser):
    args = read_args(parser)
    args = modify_args(args)
    check_args(args)
    return args


def read_args(parser):
    """
    Read all arguments specified the terminal.
    """
    parser.add_argument('--dataset', type=str, help='Dataset used for training and evaluation')
    parser.add_argument('--data_dir', type=str, default='/home/bdolicki/thesis/ssl-histo/data',
                        help='Directory of the data')
    parser.add_argument('--sample', type=float, default=None,
                        help='A fraction or number of examples that will be subsampled from the full dataset. '
                             'Useful for quick debugging and experiments with low data regimes.')
    parser.add_argument('--old_img_path_prefix', type=str,
                        help='Old path to images that will be replaced by the new prefix in the .npy files.'
                             'It is specifically useful when the .npy files where generated for one directory'
                             'and now they should be used in another.')
    parser.add_argument('--new_img_path_prefix', type=str,
                        help='New path to images that will be replaced by the new prefix.in the .npy files.'
                             'It is specifically useful when the .npy files where generated for one directory'
                             'and now they should be used in another.')
    parser.add_argument('--exp_name', type=str, default="Default",
                        help='MLFlow experiment folder where the results will be logged')
    parser.add_argument('--fold', type=int, default=None, choices=[0, 1, 2, 3, 4], help='Fold used for training and testing')
    parser.add_argument('--train_mag', type=str, choices=["40", "100", "200", "400"],
                        help='Magnitude of training images')
    parser.add_argument('--test_mag', type=str, choices=["40", "100", "200", "400"],
                        help='Magnitude of testing images')
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers used to load the data")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--model_type', type=str, default='resnet18', help='Model architecture')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help="Type of optimizer used for training")
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help="L2 penalty of the model weights that improves regularization, switched off by default")
    parser.add_argument('--lr_scheduler_type', type=str, choices=["Constant", "StepLR", "OneCycleLR", "ReduceLROnPlateau"],
                        help='Type of learning rate scheduler used for training.')
    parser.add_argument('--max_lr', type=float, default=0.001, help="Maximum learning rate")
    parser.add_argument('--start_lr', type=float, default=None, help="Initial learning rate at the start")
    parser.add_argument('--end_lr', type=float, default=None, help="Final learning rate at the end")
    parser.add_argument('--lr_pct_start', type=float, default=None,
                        help='The percentage of the cycle (in number of steps) spent increasing the learning rate.')
    parser.add_argument('--lr_warmup', type=int, default=None,
                        help='Number of epochs spent increasing the learning rate from initial learning rate to its maximum value.')
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum of stochastic gradient descent")
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of training epochs')

    parser.add_argument('--seed', type=int, default=7, help='Random seed')
    parser.add_argument('--no_validation', action="store_true",
                        help="If this argument is specified, don't evaluate on validation set")
    parser.add_argument('--no_early_stopping', action="store_true",
                        help="If this argument is specified, don't use early stopping")
    parser.add_argument('--patience', type=int, default=20, help="Number of epochs without improvement required for early stopping")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory with logs: checkpoints, parameters, metrics")
    parser.add_argument('--checkpoint_path', type=str, required=False,
                        help="Path to a pretrained model used for finetuning")
    parser.add_argument("--mlflow_dir", type=str, default="/project/bdolicki/mlflow_runs",
                        help="Directory with MLFlow logs")
    parser.add_argument("--save_model_every_n_epochs", type=int, default=1,
                        help="Defines how often the model is saved")
    parser.add_argument('--job_id', type=str, help="SLURM job id")
    parser.add_argument('--profile', action="store_true",
                        help="Use profiling to track CPU and GPU performance and memory")
    parser.add_argument('--no_rotation_transforms', action="store_true",
                        help="If this argument is specified, don't use rotations as image transformations.")
    parser.add_argument('--check_model_equivariance', action="store_true",
                        help="Check if the model is equivariant, by comparing outputs "
                             "for multiple rotations of the same image")
    # Distributed training
    parser.add_argument('--num_nodes', default=1, type=int,
                        help="Number of nodes used for training")
    parser.add_argument('--ip_address', type=str, help='ip address of the host node')
    parser.add_argument('--ngpus_per_node', default=1, type=int,
                        help='Number of gpus per node')
    # Equivariant networks
    def none_or_float(value):
        if value == 'None':
            return None
        return float(value)

    parser.add_argument('--N', type=int, default=4, help='Size of cyclic group for GCNN and maximum frequency for HNET')
    parser.add_argument('--F', type=float, default=1.0,
                        help='Frequency cut-off: maximum frequency at radius "r" is "F*r"')
    parser.add_argument('--sigma', type=float, default=0.45,
                        help='Width of the rings building the bases (std of the gaussian window)')
    parser.add_argument('--restrict', type=int, default=-1, help='Layer where to restrict SFCNN from E(2) to SE(2)')
    # FIXME not sure if sgsize argument is important here, so removed it for now
    parser.add_argument('--flip', dest="flip", action="store_true",
                        help='Use also reflection equivariance in the EXP model')
    parser.set_defaults(flip=False)
    parser.add_argument('--fixparams', dest="fixparams", action="store_true",
                        help='Keep the number of parameters of the model fixed by adjusting its topology')
    parser.set_defaults(fixparams=False)
    parser.add_argument('--deltaorth', dest="deltaorth", action="store_true",
                        help='Use delta orthogonal initialization in conv layers')
    parser.set_defaults(deltaorth=False)
    # DenseNet
    parser.add_argument('--growth_rate', type=int)
    parser.add_argument('--num_init_features', type=int)
    parser.add_argument('--last_hid_dims', type=int, default=-1,
                        help='Dimensionality of the last hidden activations for e2')


    args = parser.parse_args()
    return args


def modify_args(args):
    """
    Sometimes when an argument is not specified, it is convenient to set it based on another argument.
    This function includes all such cases.
    """
    # if start_lr and end_lr not provided, set them based on max_lr with multipliers
    # which empirically worked relatively well
    if args.start_lr is None:
        args.start_lr = args.max_lr / 10.0
    if args.end_lr is None:
        args.end_lr = args.max_lr / 100.0

    # 4 possible cases for lr_warmup and lr_pct_start
    # Case 1: both unspecified - use default lr_pct_start
    # Case 2: lr_warmup specified, lr_pct_start unspecified - use lr_pct_start = lr_warmup/num_epochs
    # Case 3: lr_warmup unspecified, lr_pct_start specified - use lr_pct_start
    # Case 4: both specified - override lr_pct_start and set lr_pct_start = lr_warmup/num_epochs
    if args.lr_warmup is not None:
        if args.lr_pct_start is not None:
            logging.warning(
                f"Both arguments `lr_warmup` and `pct_start` specified. "
                f"Setting `pct_start` as pct_start = lr_warmup/num_epochs.")
        args.lr_pct_start = args.lr_warmup / args.num_epochs
    elif args.lr_pct_start is None:
        args.lr_pct_start = 0.3

    args.early_stopping = not args.no_early_stopping
    del args.no_early_stopping

    args.multi_gpu = args.ngpus_per_node * args.num_nodes > 1
    return args


def check_args(args):
    assert not (args.early_stopping and args.no_validation), \
        "Both args.early_stopping and args.no_validation are set to True. \n" \
        "You cannot perform early stopping without evaluating the validation set. \n" \
        "Either add --no_early_stopping flag or remove --no_validation flag."

    assert args.dataset != "breakhis_fold" or (args.train_mag is not None and args.test_mag is not None)
    assert args.model_type in models, \
        f"Model {args.model_type} is not supported. Choose one of the following models: {list(models.keys())}"
    assert not args.multi_gpu or args.ip_address

def convert_model_to_single_gpu(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def add_transform_to_args(transform):
    transform_list_str = [convert_transform_to_dict(t) for t in transform.transforms]
    return transform_list_str


def add_transforms_to_args(train_transform, test_transform):
    transform = {}

    transform["train"] = add_transform_to_args(train_transform)
    transform["test"] = add_transform_to_args(test_transform)

    return transform


def check_model_equivariance(model, dataloader, device, num_classes):
    # evaluate the `model` on 8 rotated versions of the input image `x`
    model.eval()

    # take first image
    x = next(iter(dataloader))[0][0]
    ys = []

    logging.info("Probabilities of rotations of a single image")
    logging.info('#'*(num_classes+1)*6)
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(num_classes)])
    logging.info(header)
    with torch.no_grad():
        for r in range(4):
            x_transformed = F.rotate(x, r*90.).unsqueeze(dim=0) #.reshape(1, 1, img_size, img_size)
            x_transformed = x_transformed.to(device)

            y = model(x_transformed)
            y = y.to('cpu').numpy().squeeze()
            ys.append(y)

            angle = r * 90
            logging.info("{:5d} : {}".format(angle, y))

            # check the first rotation is almost equal to all other rotations
            assert np.allclose(ys[0], y, atol=1e-03), "The model is not equivariant."
    logging.info('#' * (num_classes + 2) * 6)
    logging.info("")

def parse_args_from_checkpoint(args):


    if args.checkpoint_path is not None:
        if ("supervised" in args.checkpoint_path) or ("finetune" in args.checkpoint_path):
            # read arguments of the training job
            train_log_dir = Path(args.checkpoint_path).parent.parent
            with open(os.path.join(train_log_dir, "args.json")) as json_file:
                train_args = json.load(json_file)
                train_args = Namespace(**train_args)

            # if argument not specified for evaluation, use the value from training config
            for arg in vars(train_args):
                if not hasattr(args, arg) or getattr(args, arg) is None:  # if arg doesn't exist or is None
                    setattr(args, arg, getattr(train_args, arg))

            # convert transforms from dict to list
            transform_list = [convert_dict_to_transform(t) for t in args.transform["test"]]
        else:
            train_log_dir = Path(args.checkpoint_path).parent
            with open(os.path.join(train_log_dir, "train_config.yaml")) as yaml_file:
                train_config = yaml.load(yaml_file, Loader=yaml.FullLoader)

            from vissl.data.ssl_transforms import get_transform

            # convert transforms from yaml to transform.Compose to list
            # hardcode test because it's almost always the case and a general implementation is unnecessarily complex
            test_transform = get_transform(train_config["DATA"]["TEST"]["TRANSFORMS"])
            transform_list = [t.transform for t in test_transform.transforms]

            model_family = train_config["MODEL"]["TRUNK"]["NAME"]
            model_args = train_config["MODEL"]["TRUNK"][(model_family + "s").upper()]
            args.model_type = model_family + str(model_args["DEPTH"])

            args.seed = train_config["SEED_VALUE"]

            if is_equivariant(args.model_type):
                for name, model_arg in model_args.items():
                    args.__setattr__(name, model_arg)

            args.multi_gpu = train_config["DISTRIBUTED"]["NUM_PROC_PER_NODE"] * train_config["DISTRIBUTED"]["NUM_NODES"] > 1

    else:
        default_random_init = {
                               "batch_size": 512,
                               "exp_name": "test_mre",
                               "seed": 7,
                               "model_type": "resnet18",
                               }

        # if argument not specified for evaluation, use the value from default config
        default_random_init = Namespace(**default_random_init)
        for arg in vars(default_random_init):
            if not hasattr(args, arg) or getattr(args, arg) is None:  # if arg doesn't exist or is None
                setattr(args, arg, getattr(default_random_init, arg))

        transform_list = [transforms.ToTensor()]

    transform = transforms.Compose(transform_list)
    return args, transform


def assert_pretrained_state_dict(missing_keys, unexpected_keys):
    assert missing_keys == ["fc.weight", "fc.bias"], f"Missing key(s) in state_dict: {missing_keys}"
    assert len(unexpected_keys) == 0, f"Unexpected key(s) in state_dict: {unexpected_keys}"


def init_metrics(num_classes, selected_metrics):
    metrics = {"auroc": AUROC(num_classes=num_classes, average='macro'),
               "confusion_matrix": ConfusionMatrix(num_classes=num_classes),
               "f1": F1Score(num_classes=num_classes, average='macro')}

    selected_metrics = selected_metrics.split(",")
    metrics = {k: v for k, v in metrics.items() if k in selected_metrics}
    return metrics
