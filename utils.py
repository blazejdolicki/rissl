import logging
import os
import sys
import torch
import numpy as np
import random

from models import models

def setup_logging(output_dir):
    """
    Setup various logging streams: stdout and file handlers.
    Suitable for single GPU only.
    """
    # get the filename if we want to log to the file as well
    os.makedirs(output_dir, exist_ok=True)
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
    parser.add_argument('--data_dir', type=str, default='/home/bdolicki/thesis/ssl-histo/data/breakhis',
                        help='Directory of the data')
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
    parser.add_argument('--job_id', type=str, help="SLURM job id")
    parser.add_argument('--profile', action="store_true",
                        help="Use profiling to track CPU and GPU performance and memory")
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
    return args


def check_args(args):
    assert not (args.early_stopping and args.no_validation), \
        "Both args.early_stopping and args.no_validation are set to True. \n" \
        "You cannot perform early stopping without evaluating the validation set. \n" \
        "Either add --no_early_stopping flag or remove --no_validation flag."

    assert args.dataset != "breakhis_fold" or (args.train_mag is not None and args.test_mag is not None)
    assert args.model_type in models, \
        f"Model {args.model_type} is not supported. Choose one of the following models: {list(models.keys())}"

