from ..datasamplers.mask_generators import fixed_alpha_mask_factory, CounterfactualMaskFactory
from ..datasets.test_data import RandomDataset
import numpy as np
import pytest

def test_fixed_alpha_mask_factory():
    """
    Test that the fixed_alpha_mask_factory returns the correct number of true labels.
    """
    def expected_true_count(alpha: float, labels: np.ndarray) -> int:
        """
        expected returned number of true labels with function 'fixed_alpha_mask_factory'
        """
        class_sizes = np.bincount(labels)              # assumes {0,1}
        smaller, larger = class_sizes.min(), class_sizes.max()
        per_class = int(alpha * labels.size / 2)

        if per_class < smaller:
            return per_class * 2
        elif per_class > larger:
            return smaller + larger
        else:
            return smaller + per_class

    alpha = np.random.uniform(0, 0.2)
    random_dataset = RandomDataset()
    labels = random_dataset.coarse_labels
    expected = expected_true_count(alpha, labels)
    mask = fixed_alpha_mask_factory(alpha=alpha).get_masks(labels, np.random.default_rng(0))
    assert mask.sum() == expected

def test_CounterfactualMaskFactory():
    """
    Test that the CounterfactualMaskFactory raises split valueerror as expected.
    """
    randomdata = RandomDataset()
    labels = randomdata.coarse_labels
    split = np.zeros(len(labels), dtype=bool)
    split[np.random.permutation(labels)] = np.random.randint(0, 2, np.sum(labels))
    counterfactual = CounterfactualMaskFactory(split=split, alpha=np.random.uniform(0, 0.5))

    with pytest.raises(ValueError, match="Bool split vector must index samples from one class only."):
        counterfactual.get_masks(labels, np.random.default_rng(2))

    split = np.zeros(len(labels), dtype=bool)
    split[labels] = np.random.randint(0, 2, np.sum(labels))
    counterfactual = CounterfactualMaskFactory(split=split, alpha=np.random.uniform(0, 0.5))
    assert len(counterfactual.get_masks(labels, np.random.default_rng(2))) == len(labels)
