from ..storage.training import MaskMarginStorage
from ..samplers.mask_generators import mask_factory_fixed_alpha
from ..datasets.test_data import RandomDataset
import numpy as np
from functools import partial

def test_mask_storage_masks():
    """
    Test that the masks stored in the MaskMarginStorage object are unique.
    """
    random_dataset = RandomDataset()
    storage = MaskMarginStorage(n_models=20, n_samples=random_dataset.num_samples, labels=random_dataset.coarse_labels, mask_factory=partial(mask_factory_fixed_alpha, alpha=0.1), in_memory=True)
    arr = storage.masks
    assert np.unique(arr, axis=0).shape[0] == arr.shape[0]


