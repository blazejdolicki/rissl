import argparse
from argparse import Namespace
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import logging
from torchvision import transforms
import os
from tqdm import tqdm
import mlflow
import json
import time
import copy

from datasets.breakhis_fold_dataset import BreakhisFoldDataset
from datasets.breakhis_dataset import BreakhisDataset
from datasets.pcam_dataset import PCamDataset
from utils import setup_logging, parse_args, fix_seed, convert_model_to_single_gpu
from models import models

"""
This is a very flexible script that allows evaluating a trained model on an arbitrary dataset.
It can be used both from the command line and as a function embedded into another python script. 
The latter is especially useful for aggregating the results over folds etc.
(see: scale-experiments/sup_breakhis_scale.py)
 
Example usage (command line):
```
python evaluate.py --checkpoint_path ... --split ...
```

Example usage (function):
```
from argparse import Namespace
args = {"checkpoint_path": ..., "split"}
args = Namespace(**args)
loss, acc = evaluate(args)
```

"""

def evaluate(args):
    # read arguments of the training job
    train_log_dir = Path(args.checkpoint_path).parent.parent
    with open(os.path.join(train_log_dir, "args.json")) as json_file:
        train_args = json.load(json_file)
        train_args = Namespace(**train_args)

    # if argument not specified for evaluation, use the value from training config
    for arg in vars(train_args):
        if not hasattr(args, arg) or getattr(args, arg) is None: # if arg doesn't exist or is None
            setattr(args, arg, getattr(train_args, arg))

    job_id = args.log_dir.split("/")[-1]
    mlflow.set_tracking_uri(f"file:///{args.mlflow_dir}")
    mlflow.set_experiment(args.exp_name)
    mlflow.start_run(run_name=job_id)
    mlflow_args = copy.deepcopy(args)
    mlflow_args.transform = "See args.json"
    mlflow.log_params(vars(mlflow_args))
    setup_logging(args.log_dir)

    fix_seed(args.seed)

    # same transforms as for supervised equivariant networks
    transform = transforms.Compose([transforms.ToTensor()])

    transform_list_str = [str(t) for t in transform.transforms]
    assert transform_list_str == args.transform["test"], \
        "Current image transformations are different than those used for evaluation during training"

    args_path = os.path.join(args.log_dir, "args.json")
    with open(args_path, 'w') as file:
        json.dump(vars(args), file, indent=4)

    if args.dataset == "breakhis_fold":
        dataset = BreakhisFoldDataset(args.data_dir, args.split, args.fold, args.test_mag, transform)
    elif args.dataset == "breakhis":
        dataset = BreakhisDataset(args.data_dir, args.split, transform)
    elif args.dataset == "pcam":
        dataset = PCamDataset(root_dir=args.data_dir, split=args.split, transform=transform)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # select a model with randomly initialized weights, default is resnet18 so that we can train it quickly
    model_args = {"num_classes": 2}  # CE loss
    densenet = {"growth_rate": args.growth_rate, "block_config": (3, 3, 3), "num_init_features": args.num_init_features}
    model_args = {**model_args, **densenet}
    model = models[train_args.model_type](**model_args).to(device)

    # load model from checkpoint
    start_time = time.time()

    state_dict = torch.load(args.checkpoint_path)

    if args.multi_gpu:
        state_dict = convert_model_to_single_gpu(state_dict)

    model.load_state_dict(state_dict)
    end_time = time.time()
    logging.debug(f"Loading the model took {end_time - start_time} seconds")

    criterion = torch.nn.CrossEntropyLoss()

    epoch_loss = 0.0
    correct = 0.0
    actual_data_size = 0.0
    with torch.no_grad():
        for batch in dataloader:
            model.eval()

            inputs, labels = batch
            batch_size = inputs.shape[0]

            inputs = inputs.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)

            epoch_loss += batch_size * batch_loss.item()
            actual_data_size += batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    epoch_loss = epoch_loss / actual_data_size
    epoch_acc = 100 * correct / actual_data_size

    logging.info(f"Test loss: \t{epoch_loss}, test acc: \t{epoch_acc}")

    mlflow.log_metric("loss", epoch_loss)
    mlflow.log_metric("acc", epoch_acc)

    mlflow.end_run()

    return epoch_loss, epoch_acc

if __name__ == "__main__":
    # in this scripts the default values are usually the values from the training script
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset used for evaluation')
    parser.add_argument('--data_dir', type=str, help='Directory of the data')
    parser.add_argument('--split', type=str, required=True, choices=["train", "test", "valid"])
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to a trained model used for evaluation")
    parser.add_argument('--exp_name', type=str,
                        help='MLFlow experiment folder where the results will be logged')
    parser.add_argument('--fold', type=int, choices=[1, 2, 3, 4, 5],
                        help='Fold used for training and testing')
    parser.add_argument('--test_mag', type=str, choices=["40", "100", "200", "400"],
                        help='Magnitude of testing images')
    parser.add_argument('--num_workers', type=int, help="Number of workers used to load the data")
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument("--log_dir", type=str, default="logs_eval",
                        help="Directory with logs: checkpoints, parameters, metrics")
    parser.add_argument("--mlflow_dir", type=str, default="/project/bdolicki/mlflow_runs",
                        help="Directory with MLFlow logs")
    args = parser.parse_args()

    loss, acc = evaluate(args)