from ..storage.splits import SplitStorage, Split
from ..storage.registry import RandomDataset

import numpy as np

def test_SplitStorage():
    """
    test that the initial split in SplitStorage indexes either class.
    """
    dataset = RandomDataset()
    coarse_labels = dataset.coarse_labels
    IDs = np.argwhere(coarse_labels).reshape(-1)
    split_storage = SplitStorage(splits={'split_0':Split(IDs)})

    assert len(split_storage.splits['split_0'].data)==coarse_labels.sum()
    assert split_storage.splits['split_0'].finished is False

    coarse_labels = np.invert(dataset.coarse_labels)
    IDs = np.argwhere(coarse_labels).reshape(-1)
    split_storage = SplitStorage(splits={'split_0':Split(IDs)})

    assert len(split_storage.splits['split_0'].data)==coarse_labels.sum()


def test_return_counterfactual_basic():
    ...

def test_pipeline_cfl():
    ...

def test_return_best_split_interface():
    ...

def test_return_best_spectral_split():
    ...

def test_spectral_split_weights():
    ...

def test_pipeline_split():
    ...