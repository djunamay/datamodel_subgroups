import chz
from .base import MaskFactory, MaskFactoryInitializer
import numpy as np
from numpy.typing import NDArray

@chz.chz
class fixed_alpha_mask_factory(MaskFactory):
    """
    Generates balanced binary masks for training sets, selecting a fixed proportion (`alpha`) 
    of samples. Half of these samples are selected from each class to maintain balance.

    Attributes
    ----------
    alpha : float
        Proportion of the total dataset to include in the training mask (0 < alpha ≤ 1).
        Default is 0.5.
    """
    alpha: float = chz.field(default=0.5, doc='Proportion of the total dataset to include in the training mask (0 < alpha ≤ 1). Half of these samples are selected from each class to maintain balance.')
    min_test_fraction: float = chz.field(default=0.3)

    
    def get_masks(self, labels: NDArray[bool], rng: np.random.Generator) -> NDArray[bool]:
        """
        Create a balanced mask selecting samples equally from both classes, based on the `alpha` value.

        Parameters
        ----------
        labels : NDArray[bool]
            Binary array of class labels (`True` for positive class, `False` for negative).
        rng : np.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        NDArray[bool]
            Binary mask indicating selected samples (`True`) for training.
        """
        samples_per_class = int((len(labels) * self.alpha) / 2)
        indices_class_0 = np.where(~labels)[0]
        indices_class_1 = np.where(labels)[0]

        # sanity check 
        max_class_size = min(len(indices_class_0), len(indices_class_1))
        if samples_per_class > (max_class_size * (1 - self.min_test_fraction)):
            raise ValueError(
                f"Cannot sample {samples_per_class} per class: "
                f"only {len(indices_class_1)} positives and "
                f"{len(indices_class_0)} negatives available."
            )
    
        mask = np.zeros(len(labels), dtype=bool)
        mask[rng.permutation(indices_class_1)[:samples_per_class]] = True
        mask[rng.permutation(indices_class_0)[:samples_per_class]] = True
        return mask

@chz.chz
class fixed_alpha_mask_factory_initializer(MaskFactoryInitializer):
    """
    Initializes a fixed_alpha_mask_factory with a random alpha value between lower_bound and upper_bound.

    Attributes
    ----------
    lower_bound : float
        Lower bound for the alpha value.
    upper_bound : float
        Upper bound for the alpha value.
    """
    lower_bound: float = chz.field(default=0.0, doc='Lower bound for the alpha value.')
    upper_bound: float = chz.field(default=0.5, doc='Upper bound for the alpha value.')

    def build_mask_factory(self, rng: np.random.Generator) -> MaskFactory:
        alpha = rng.uniform(self.lower_bound, self.upper_bound)
        return fixed_alpha_mask_factory(alpha=alpha)