import itertools
import random
import numpy as np


def sample_runs(num_runs, *hyperparams):
    """
    Randomly select hyperparameter sets that will be used in the search
    :param hyperparams: list of lists with potential values of different hyperparameters
    :return: randomly selected hyperparameter sets
    """
    # itertools.product() returns all combinations of provided hyperparams
    all_runs = np.array(list(itertools.product(**hyperparams)))
    # select n random hyperparam sets
    selected_run_idxs = random.sample(range(len(all_runs)), num_runs)
    selected_runs = all_runs[selected_run_idxs]
    return selected_runs
