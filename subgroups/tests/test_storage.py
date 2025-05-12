from ..datastorage.base import MaskMarginStorage
from ..datasamplers.mask_generators import fixed_alpha_mask_factory
from ..datasets.test_data import RandomDataset
import numpy as np

def test_mask_storage_masks():
    """
    Test that the masks stored in the MaskMarginStorage object are unique.
    """
    random_dataset = RandomDataset()
    storage = MaskMarginStorage(n_models=20, n_samples=random_dataset.num_samples, labels=random_dataset.coarse_labels, mask_factory=fixed_alpha_mask_factory(alpha=0.1), in_memory=True)
    arr = storage.masks
    assert np.unique(arr, axis=0).shape[0] == arr.shape[0]


