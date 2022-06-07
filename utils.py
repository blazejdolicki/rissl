import logging
import os
import sys
import torch
import numpy as np
import random
import mlflow
import copy

from models import models
from collect_env import collect_env_info


def setup_logging(output_dir):
    """
    Setup various logging streams: stdout and file handlers.
    Suitable for single GPU only.
    """
    # get the filename if we want to log to the file as well
    log_filename = f"{output_dir}/log.txt"
    logging.basicConfig(
        level=logging.DEBUG,
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
    parser.add_argument('--fold', type=int, default=1, choices=[1, 2, 3, 4, 5], help='Fold used for training and testing')
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
    parser.add_argument('--lr_scheduler_type', type=str, choices=["StepLR", "OneCycleLR"],
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
    parser.add_argument("--mlflow_dir", type=str, default="/project/bdolicki/mlflow_runs",
                        help="Directory with MLFlow logs")
    parser.add_argument("--save_model_every_n_epochs", type=int, default=1,
                        help="Defines how often the model is saved")
    parser.add_argument('--job_id', type=str, help="SLURM job id")
    parser.add_argument('--profile', action="store_true",
                        help="Use profiling to track CPU and GPU performance and memory")
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
    parser.add_argument('--conv2triv', action="store_true",
                        help='Convert to trivial representation in last layer to obtain invariant outputs. If False,'
                             'use GroupPooling instead.')
    parser.set_defaults(conv2triv=False)
    parser.add_argument('--deltaorth', dest="deltaorth", action="store_true",
                        help='Use delta orthogonal initialization in conv layers')
    # DenseNet
    parser.add_argument('--growth_rate', type=int)
    parser.add_argument('--num_init_features', type=int)

    parser.set_defaults(deltaorth=False)
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


def add_transform_to_args(train_transform, test_transform):
    transform = {}

    train_transform_list_str = [str(t) for t in train_transform.transforms]
    transform["train"] = train_transform_list_str

    test_transform_list_str = [str(t) for t in test_transform.transforms]
    transform["test"] = test_transform_list_str
    return transform


