import numpy as np
from numpy.typing import NDArray
from typing import Callable

select_samples_fn = Callable[[int], NDArray[int]]

def select_features_sequential(samples_per_batch, batch_starter_seed)-> NDArray[int]:
    """
    Basic feature selection function.
    Arguments
    ---------
    n_features: int
        Number of features to select for training
    Returns
    -------
    NDArray[int]
        indices of selected features
    """
    start = batch_starter_seed * samples_per_batch
    end = start + samples_per_batch
    return list(range(start, end))