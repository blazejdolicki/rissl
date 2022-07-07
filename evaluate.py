import argparse
import torch
from torch.utils.data import DataLoader
import logging
import os
import mlflow
import json

from datasets.breakhis_fold_dataset import BreakhisFoldDataset
from datasets.breakhis_dataset import BreakhisDataset
from datasets.pcam_dataset import PCamDataset
import utils
from models import get_model

"""
This is a very flexible script that allows evaluating a trained model on an arbitrary dataset.
It can be used both from the command line and as a function embedded into another python script. 
The latter is especially useful for aggregating the results over folds etc.
(see: scale-experiments/sup_breakhis_scale.py)
 
Example usage (command line):
```
python evaluate.py --checkpoint_path ... --splits ...
```

Example usage (function):
```
from argparse import Namespace
args = {"checkpoint_path": ..., "splits"}
args = Namespace(**args)
results = evaluate(args)
```

"""


def evaluate(args):
    # Our logging needs to be defined before importing libraries that set up their own logging such as get_transform()
    # https://stackoverflow.com/questions/20240464/python-logging-file-is-not-working-when-using-logging-basicconfig
    os.makedirs(args.log_dir, exist_ok=True)
    utils.setup_logging(args.log_dir)

    args, transform = utils.parse_args_from_checkpoint(args)

    print("transforms:")
    print(transform)

    utils.fix_seed(args.seed)

    utils.setup_mlflow(args)

    args_path = os.path.join(args.log_dir, "args.json")
    with open(args_path, 'w') as file:
        json.dump(vars(args), file, indent=4)

    splits = args.splits.split(",")
    assert_splits(splits)

    # evaluate one or multiple splits
    results = {}
    for split in splits:
        results[split] = evaluate_split(args, split, transform)

    return results


def evaluate_split(args, split, transform):
    if args.dataset == "breakhis_fold":
        num_classes = 2
        dataset = BreakhisFoldDataset(args.data_dir, split, args.fold, args.test_mag, transform)
    elif args.dataset == "breakhis":
        num_classes = 2
        dataset = BreakhisDataset(root_dir=args.data_dir, split=split, transform=transform)
    elif args.dataset == "pcam":
        num_classes = 2
        dataset = PCamDataset(root_dir=args.data_dir, split=split, transform=transform, sample=args.sample)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # select a model with randomly initialized weights, default is resnet18 so that we can train it quickly
    model = get_model(args.model_type, num_classes, args).to(device)

    # load model from checkpoint
    logging.info(f"Loading model {args.model_type} from {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path)

    if args.multi_gpu:
        state_dict = utils.convert_model_to_single_gpu(state_dict)

    model.load_state_dict(state_dict)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    epoch_loss = 0.0
    correct = 0.0
    actual_data_size = 0.0
    with torch.no_grad():
        for batch in dataloader:


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

    logging.info(f"{split} loss: \t{epoch_loss}, {split} acc: \t{epoch_acc}")

    mlflow.log_metric(f"{split}_loss", epoch_loss)
    mlflow.log_metric(f"{split}_acc", epoch_acc)

    mlflow.end_run()

    return {"loss": epoch_loss, "acc": epoch_acc}

def assert_splits(splits):
    possible_splits = ["test", "val", "valid"]
    for split in splits:
        assert split in possible_splits, f"Invalid split name: {split}. Possible splits are: {possible_splits}."

if __name__ == "__main__":
    # in this scripts the default values are usually the values from the training script
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset used for evaluation')
    parser.add_argument('--data_dir', type=str, help='Directory of the data')
    parser.add_argument('--splits', type=str, required=True,
                        help="One or multiple comma-delimted splits. Usually we use valid and test (so 'valid,test'), "
                             "to double check that the validation metrics overlap with the numbers from training logs "
                             "and to obtain the test metrics.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to a trained model used for evaluation")
    parser.add_argument('--exp_name', type=str, default="Default",
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
    parser.add_argument('--sample', type=float, default=None,
                        help='A fraction or number of examples that will be subsampled from the full dataset. '
                             'Useful for quick debugging and experiments with low data regimes.')

    args = parser.parse_args()

    results = evaluate(args)

    results_path = os.path.join(args.log_dir, "results.json")
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4)