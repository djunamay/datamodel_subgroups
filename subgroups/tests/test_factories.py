from ..datasamplers.mask_generators import fixed_alpha_mask_factory
from ..datasets.test_data import RandomDataset
import numpy as np

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

    alpha = np.random.uniform(0, 1)
    random_dataset = RandomDataset()
    labels = random_dataset.coarse_labels
    expected = expected_true_count(alpha, labels)
    mask = fixed_alpha_mask_factory(alpha=alpha).get_masks(labels)
    assert mask.sum() == expected