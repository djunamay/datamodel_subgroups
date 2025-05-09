from typing import Protocol
from numpy.typing import NDArray
import numpy as np
import chz

@chz.chz
class MaskFactory():
    """
    Interface for generating boolean masks to select training samples based on provided labels.

    Implementations of this class generate masks indicating the subset of samples to be used for training,
    ensuring flexibility in how subsets are chosen (e.g., balanced, random, or stratified).
    """

    def get_masks(self, labels: NDArray[bool], seed: int = None) -> NDArray[bool]:
        """
        Generate a boolean mask to select samples for training.

        Parameters
        ----------
        labels : NDArray[bool]
            Binary array indicating class membership of samples (`True` or `False`).
        seed : int, optional
            Random seed for reproducibility. Defaults to None.

        Returns
        -------
        NDArray[bool]
            Boolean mask array, where `True` marks samples selected for training.
        """
        ...
