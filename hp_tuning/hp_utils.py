import itertools
import random
import numpy as np
import os
import json


def sample_runs(num_runs, *hyperparams):
    """
    Randomly select hyperparameter sets that will be used in the search
    :param hyperparams: list of lists with potential values of different hyperparameters
    :return: randomly selected hyperparameter sets
    """
    # set fixed seed for sampling runs
    seed = 7
    random.seed(seed)

    # itertools.product() returns all combinations of provided hyperparams
    all_runs = np.array(list(itertools.product(*hyperparams)))
    # select n random hyperparam sets
    selected_run_idxs = random.sample(range(len(all_runs)), num_runs)
    selected_runs = all_runs[selected_run_idxs]
    return selected_runs


def save_runs(search, log_dir, num_runs, runs):
    search_args = {}
    search_args["all_runs"] = search
    search_args["num_runs"] = num_runs
    search_args["selected_runs"] = []
    for run in runs:
        run_args = {}
        for i, hp in enumerate(search):
            run_args[hp] = run[i]
        search_args["selected_runs"].append(run_args)

    args_path = os.path.join(log_dir, "hp_args.json")
    with open(args_path, 'w') as file:
        json.dump(search_args, file, indent=4)