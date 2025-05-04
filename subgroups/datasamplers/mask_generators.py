import chz
from .base import MaskFactory
import numpy as np
from numpy.typing import NDArray

@chz.chz
class fixed_alpha_mask_factory(MaskFactory):
    """
    This mask factory generates a mask with a fixed alpha. An equal number of samples from each class are selected for training.
    """
    alpha: float = chz.field(default=0.5, doc='The alpha parameter for the mask generator.')
    seed: int = chz.field(default=None, doc='The seed used for the random number generator.')

    @chz.init_property
    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)
    
    def get_masks(self, labels: NDArray[bool]) -> NDArray[bool]:
        samples_per_class = int((len(labels)*self.alpha)/2)
        indices_class_0 = np.where(~labels)[0]
        indices_class_1 = np.where(labels)[0]
        mask = np.zeros(len(labels), dtype=bool)
        mask[self._rng.permutation(indices_class_1)[:samples_per_class]] = True
        mask[self._rng.permutation(indices_class_0)[:samples_per_class]] = True
        return mask