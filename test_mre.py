import argparse
from argparse import Namespace
import os

from evaluate_mre import evaluate_mre
import utils

if __name__ == "__main__":

    # TODO this probably can be refactored to avoid repeating code from evaluate_mre.py
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='Dataset used for evaluation')
    parser.add_argument('--data_dir', type=str, help='Directory of the data')
    parser.add_argument('--split', type=str, choices=["train", "test", "valid"])
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="Path to a trained model used for evaluation. "
                             "If not specified, use a randomly initialized model")
    parser.add_argument('--exp_name', type=str,
                        help='MLFlow experiment folder where the results will be logged')
    parser.add_argument('--fold', type=int, choices=[1, 2, 3, 4, 5],
                        help='Fold used for training and testing')
    parser.add_argument('--test_mag', type=str, choices=["40", "100", "200", "400"],
                        help='Magnitude of testing images')
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers used to load the data")
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument("--log_dir", type=str, default="logs_eval",
                        help="Directory with logs: checkpoints, parameters, metrics")
    parser.add_argument("--mlflow_dir", type=str, default=os.path.join(os.getcwd(),"mlflow_runs"),
                        help="Directory with MLFlow logs")
    parser.add_argument('--sample', type=float, default=None,
                        help='A fraction or number of examples that will be subsampled from the full dataset. '
                             'Useful for quick debugging and experiments with low data regimes.')
    parser.add_argument("--mre_n", type=int, default=4,
                        help='Number of rotations N sampled when evaluating MRE(N)')
    parser.add_argument("--save_samples", type=int, default=0,
                        help='Number of batches for which we plot the rotated images together with their probabilities.'
                             'By default, no images are saved.')

    # split is not important for dummy dataset, but it's required argument
    args = parser.parse_args(namespace=Namespace(split="test"))

    args.dataset = "dummy"

    assert evaluate_mre(args) == 0.0

    print("Test passed")