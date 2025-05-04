from typing import Protocol
from numpy.typing import NDArray
import numpy as np
import chz

@chz.chz
class MaskFactory():
    """
    A factory that returns a boolean mask of the same length as the labels, where the True values are the indices of the samples to be used for training.
    """

    def get_masks(self, labels: NDArray[bool]) -> NDArray[bool]:
        ...
