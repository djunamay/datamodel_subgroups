import numpy as np
from numpy.typing import NDArray
from typing import Callable
from functools import partial

mask_factory_fn = Callable[[NDArray[bool], np.random.Generator], NDArray[bool]]
"""Callable signature for mask functions.
Parameters
----------
labels : NDArray[bool]
    Boolean class labels.
rng : np.random.Generator
    Random generator used to sample a mask.
Returns
-------
NDArray[bool]
    Mask of indices for training.
"""

def mask_factory_fixed_alpha(labels: NDArray[bool], rng: np.random.Generator, alpha: float=0.1, min_test_fraction: float=0.3) -> NDArray[bool]:
    """
    Generates balanced binary masks for training sets, selecting a fixed proportion (`alpha`)
    of samples. Half of these samples are selected from each class to maintain balance.

    Attributes
    ----------
    alpha : float
        Proportion of the total dataset to include in the training mask (0 < alpha ≤ 1).
        Default is 0.5.
    min_test_fraction : float
        Required minimum test fraction user wants to allow for.
    """
    samples_per_class = int((len(labels) * alpha) / 2)
    indices_class_0 = np.where(~labels)[0]
    indices_class_1 = np.where(labels)[0]

    # sanity check
    max_class_size = min(len(indices_class_0), len(indices_class_1))
    if samples_per_class > (max_class_size * (1 - min_test_fraction)):
        raise ValueError(
            f"Cannot sample {samples_per_class} per class: "
            f"only {len(indices_class_1)} positives and "
            f"{len(indices_class_0)} negatives available."
        )

    mask = np.zeros(len(labels), dtype=bool)
    mask[rng.permutation(indices_class_1)[:samples_per_class]] = True
    mask[rng.permutation(indices_class_0)[:samples_per_class]] = True
    return mask


def mask_factory_counterfactuals(labels: NDArray[bool], rng: np.random.Generator, split: NDArray[bool], alpha: float=0.1, min_test_fraction: float=0.3) -> NDArray[bool]:
    """
    Generates balanced binary masks for training sets, selecting a fixed proportion (`alpha`)
    of samples. Half of these samples are selected from each class to maintain balance. For one of the classes sampling is restricted to a subset of samples for that class, as indicated by the split vector.

    Attributes
    ----------
    alpha : float
        Proportion of the total dataset to include in the training mask (0 < alpha ≤ 1).
        Default is 0.5.
    split: NDArray[bool]
        Boolean vector of length num_samples which splits class 0 OR class 1 samples in two.
    min_test_fraction: float
        Requred minimum test fraction user wants to allow for. If specified alpha is too large to allow for that test fraction, ValueError is raised.
    """

    # check that split vector only indexes samples from a single class
    labels_in_split, labels_out_split = np.unique(labels[split]), np.unique(labels[~split])
    split_ok = (len(labels_in_split) == 1) & (len(labels_out_split) == 2)

    if not split_ok:
        raise ValueError('Bool split vector must index samples from one class only.')

    samples_per_class = int((len(labels) * alpha) / 2)

    if not labels_in_split:  # if self.split does not index the 1's class, invert labels, otherwise subsequent indices for class 0 and 1 will both point to class 0
        labels = np.invert(labels)

    indices_class_0 = np.where(~labels)[0]
    indices_class_1 = np.where(split)[0]

    # sanity check
    max_class_size = min(len(indices_class_0), len(indices_class_1))
    if samples_per_class > (max_class_size * (1 - min_test_fraction)):
        raise ValueError(
            f"Cannot sample {samples_per_class} per class: "
            f"only {len(indices_class_1)} positives and "
            f"{len(indices_class_0)} negatives available."
        )

    mask = np.zeros(len(labels), dtype=bool)
    mask[rng.permutation(indices_class_1)[:samples_per_class]] = True
    mask[rng.permutation(indices_class_0)[:samples_per_class]] = True

    return mask


mask_factory_init_fn = Callable[[np.random.Generator], mask_factory_fn]
"""Callable signature for mask function generators.
Parameters
----------
rng : np.random.Generator
    Random generator used to sample new mask_factory_fn
Returns
-------
mask_factory_fn
    A callable with signature (labels, rng) -> mask, where `labels` is
    a boolean array and `rng` is a NumPy random generator.
"""

def mask_factory_init_fixed_alpha(rng: np.random.Generator, lower_bound: int=0.1, upper_bound: int=0.5) -> mask_factory_fn:
    """
    Initializes a fixed_alpha_mask_factory with a random alpha value between lower_bound and upper_bound with fixed min test size.

    Attributes
    ----------
    lower_bound : float
        Lower bound for the alpha value.
    upper_bound : float
        Upper bound for the alpha value.
    """
    alpha = rng.uniform(lower_bound, upper_bound)
    return partial(mask_factory_fixed_alpha, alpha=alpha)
