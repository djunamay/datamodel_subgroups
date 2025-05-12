from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Union, Optional
import numpy as np
from abc import abstractmethod, ABC
from subgroups.datasamplers.base import MaskFactory
from numpy.typing import NDArray
import warnings
import os
Array = Union[np.ndarray, np.memmap]

class MaskMarginStorageInterface(ABC):
    """
    Interface for managing storage of masks and margins for multiple models.

    This interface defines the structure for storing and accessing masks, margins, 
    and test accuracies associated with a set of models. It ensures that implementations 
    provide methods to check and fill storage for individual model instances.
    """
    @property
    @abstractmethod
    def masks(self) -> Array:
        """Masks array (shape: [n_models, n_samples])."""
        ...
    
    @property
    @abstractmethod
    def margins(self) -> Array:
        """Margins array (shape: [n_models, n_samples])."""
        ...

    @property
    @abstractmethod
    def test_accuracies(self) -> Array:
        """Test accuracies array (shape: [n_models])."""
        ...
    
    @abstractmethod
    def is_filled(self, instance_index: int) -> bool:
        """
        Check if the storage for a specific model instance is filled.

        Parameters
        ----------
        instance_index : int
            Index of the model instance to check.

        Returns
        -------
        bool
            True if the instance is filled, False otherwise.
        """
        ...

    @abstractmethod
    def fill_results(self, instance_index: int, margins: Array, test_accuracy: float):
        """
        Fill the storage with results for a specific model instance.

        Parameters
        ----------
        instance_index : int
            Index of the model instance to fill.
        margins : Array
            Margins array for the model instance.
        test_accuracy : float
            Test accuracy for the model instance.
        """
        ...
