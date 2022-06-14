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
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


from datasets.breakhis_fold_dataset import BreakhisFoldDataset
from datasets.breakhis_dataset import BreakhisDataset
from datasets.pcam_dataset import PCamDataset
from datasets.dummy_dataset import DummyDataset
from datasets.discrete_rotation import DiscreteRotation

import utils
from models import get_model

"""
This is a very flexible script that allows evaluating the Mean Rotation Error (MRE) of 
a trained model on an arbitrary dataset.
It can be used both from the command line and as a function embedded into another python script. 
The latter is especially useful for aggregating the results over folds etc.
(see: scale-experiments/sup_breakhis_scale.py)

Example usage (command line):
```
python evaluate_mre.py --checkpoint_path ... --split ...
```

Example usage (function):
```
from argparse import Namespace
args = {"checkpoint_path": ..., "split"}
args = Namespace(**args)
loss, acc = evaluate_mre(args)
```

"""

def evaluate_mre(args):
    if args.checkpoint_path is not None:
        # read arguments of the training job
        train_log_dir = Path(args.checkpoint_path).parent.parent
        with open(os.path.join(train_log_dir, "args.json")) as json_file:
            train_args = json.load(json_file)
            train_args = Namespace(**train_args)

        # if argument not specified for evaluation, use the value from training config
        for arg in vars(train_args):
            if not hasattr(args, arg) or getattr(args, arg) is None:  # if arg doesn't exist or is None
                setattr(args, arg, getattr(train_args, arg))
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

    os.makedirs(args.log_dir, exist_ok=True)

    utils.setup_logging(args.log_dir)

    utils.fix_seed(args.seed)

    # here we have to set the transforms manually because reading them from saved config is not trivial to implement
    transform = transforms.Compose([transforms.ToTensor()])

    transform_list_str = [str(t) for t in transform.transforms]
    if args.checkpoint_path is not None:
        assert transform_list_str == args.transform["test"], \
            "Current image transformations are different than those used for evaluation during training"

    utils.setup_mlflow(args)

    args_path = os.path.join(args.log_dir, "args.json")
    with open(args_path, 'w') as file:
        json.dump(vars(args), file, indent=4)

    if args.dataset == "breakhis_fold":
        num_classes = 4
        dataset = BreakhisFoldDataset(args.data_dir, args.split, args.fold, args.test_mag, transform)
    elif args.dataset == "breakhis":
        num_classes = 4
        dataset = BreakhisDataset(args.data_dir, args.split, transform)
    elif args.dataset == "pcam":
        num_classes = 2
        dataset = PCamDataset(root_dir=args.data_dir, split=args.split, transform=transform, sample=args.sample)
    elif args.dataset == "dummy":
        num_classes = 2
        dataset = DummyDataset(num_classes=num_classes)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # select a model with randomly initialized weights, default is resnet18 so that we can train it quickly
    model = get_model(args.model_type, num_classes, args).to(device)
    if args.checkpoint_path is not None:
        logging.debug(f"Loading model {args.model_type} from {args.checkpoint_path}")
        # load model from checkpoint
        state_dict = torch.load(args.checkpoint_path)

        if args.multi_gpu:
            state_dict = utils.convert_model_to_single_gpu(state_dict)

        model.load_state_dict(state_dict)
    else:
        logging.debug(f"Model {args.model_type} initialized from scratch.")

    N = args.mre_n
    angles = np.linspace(start=0, stop=360, num=N, endpoint=False)
    rotations = [DiscreteRotation(angles=[angle]) for angle in angles]

    stds = []
    with torch.no_grad():
        for batch in dataloader:
            model.eval()

            # shape of inputs: (B, C, H, W)
            inputs, labels = batch
            # number of unique images in the batch
            batch_size = inputs.shape[0]

            rotated_inputs = [rotation(inputs) for rotation in rotations]

            # just for debugging: save first image in the batch, permute from (C, H, W) to (H, W, C) and save image
            # for i, angle in enumerate(angles):
            #     plt.imsave(f'1st image - {angle} degrees.png', np.asarray(rotated_inputs[i][3].permute(1,2,0)))

            # concatenated list of N tensors of shape (B, C, H, W) into a single (N*B, C, H, W) tensor
            # first B indices in the first dimension relate to all images in batch with the first rotation
            rotated_inputs = torch.cat(rotated_inputs, dim=0)

            rotated_inputs = rotated_inputs.to(device)
            labels = labels.to(device)

            # calculate outputs by passing images through the network
            outputs = model(rotated_inputs)

            # obtain probabilties from the outputs
            probs = F.softmax(outputs, dim=1)

            #  reshape to (B, N, num_classes) in order to have separate dimensions for images and rotations
            probs = probs.reshape((N, batch_size, -1)).permute(1, 0, 2)

            # reshape labels to select the correct class probability in gather()
            label_indices = labels.unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, N, 1)

            # select probabilities of the correct classes per image
            target_probs = torch.gather(probs, 2, label_indices) # shape: (B, N, 1)

            # compute standard deviations between rotations of each image and add it to the list
            batch_stds = torch.std(target_probs, dim=1).squeeze()
            # print("batch probs", probs)
            # print("batch stds", batch_stds.tolist())
            stds += batch_stds.tolist()
    # print("stds", stds)
    mre = np.mean(stds)

    logging.info(f"MRE({N}): \t{mre}")
    mlflow.log_metric(f"MRE_{N}", mre)

    mlflow.end_run()

    return mre


if __name__ == "__main__":
    # in this scripts the default values are usually the values from the training script
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='Dataset used for evaluation')
    parser.add_argument('--data_dir', type=str, help='Directory of the data')
    parser.add_argument('--split', type=str, required=True, choices=["train", "test", "valid"])
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="Path to a trained model used for evaluation. "
                             "If not specified, use a randomly initialized model")
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
    parser.add_argument("--mlflow_dir", type=str, default="mlflow_runs",
                        help="Directory with MLFlow logs")
    parser.add_argument('--sample', type=float, default=None,
                        help='A fraction or number of examples that will be subsampled from the full dataset. '
                             'Useful for quick debugging and experiments with low data regimes.')
    parser.add_argument("--mre_n", type=int, default=4,
                        help='Number of rotations N sampled when evaluating MRE(N)')
    args = parser.parse_args()

    mre = evaluate_mre(args)