import chz
import numpy as np
from numpy.typing import NDArray
from typing import Callable

select_features_fn = Callable[[int], NDArray[int]]

def select_features_basic(n_features: int)-> NDArray[int]:
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
    return np.arange(n_features)

def select_features_singlecell(n_features: int, n_celltypes: int, n_total: int, n_stats: int):
    """
    Select feature indices from a block-structured single-cell representation.

    The dataset is assumed to be arranged in the following **column layout**:

        [ celltype 0 block | celltype 1 block | ... | celltype (n_celltypes-1) block ]

    Each cell-type block contains:
        n_total * n_stats columns,
    corresponding to ``n_total`` features (e.g. PCs), each expanded into ``n_stats`` statistics.

    From each cell-type block, this function selects the first::

        n_features * n_stats

    columns (i.e. the first ``n_features`` features, across all statistics).

    Parameters
    ----------
    n_features : int
        Number of features to select per cell type (before expansion by ``n_stats``).
    n_celltypes : int
        Number of cell types in the dataset.
    n_total : int
        Total number of features (e.g. PCs) available per cell type in the dataset.
    n_stats : int
        Number of statistics computed for each feature.

    Returns
    -------
    np.ndarray of int
        A 1D array of column indices corresponding to the selected features across
        all cell-type blocks. Length is ``n_celltypes * n_features * n_stats``.
    """
    features_sele = np.arange(n_features * n_stats)
    features_per_celltype = n_total * n_stats
    return np.hstack([features_sele + (features_per_celltype * celltype) for celltype in range(n_celltypes)])

