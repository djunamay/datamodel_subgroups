import os
import warnings
from typing import Optional
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from .base import MaskMarginStorageInterface
from ..datasamplers import MaskFactory
from typing import Union
Array = Union[np.ndarray, np.memmap]
from functools import cached_property
@dataclass
class MaskMarginStorage(MaskMarginStorageInterface):
    """
    Storage class for managing masks, margins, and test accuracies for multiple models.

    Attributes
    ----------
    n_models : int
        Number of models.
    n_samples : int
        Number of samples.
    labels : NDArray[bool]
        Binary labels for the samples.
    mask_factory : MaskFactory
        Factory to generate masks.
    in_memory : bool, optional
        Flag to determine if data is stored in memory. Default is True.
    path : Optional[str], optional
        Path for memory-mapped files if not in memory. Default is None.
    rng : Optional[np.random.Generator], optional
        Random number generator for mask generation. Default is None.
    """
    n_models: int
    n_samples: int
    labels: NDArray[bool]
    mask_factory: MaskFactory
    in_memory: bool = True
    path: Optional[str] = None   
    rng: Optional[np.random.Generator] = np.random.default_rng(0)
    batch_starter_seed: int = 0

    def __post_init__(self):
        """
        Validate initialization parameters and issue warnings if necessary.
        """
        if not self.in_memory and self.path is None:
            raise ValueError(
                "in_memory=False requires a valid 'path' for the memory‑mapped files"
            )

        if self.in_memory and self.path is not None:
            warnings.warn(
                "The supplied 'path' will be ignored because in_memory=True",
                UserWarning,
            )

    @staticmethod
    def _create_array(in_memory: bool, path: Optional[str], dtype: np.dtype, shape: tuple[int, int]) -> Array:
        """
        Create an array either in memory or as a memory-mapped file.

        Parameters
        ----------
        in_memory : bool
            Flag to determine if the array is stored in memory.
        path : Optional[str]
            Path for memory-mapped file if not in memory.
        dtype : np.dtype
            Data type of the array.
        shape : tuple[int, int]
            Shape of the array.

        Returns
        -------
        Array
            Initialized array.
        """
        if in_memory:
            return np.zeros(shape, dtype=dtype)

        if path is None:
            raise ValueError("path must be provided when in_memory=False")

        mode = "r+" if os.path.exists(path) else "w+"
        return np.lib.format.open_memmap(path, dtype=dtype, mode=mode, shape=shape)

        #return np.memmap(path, dtype=dtype, mode=mode, shape=shape)

    
    @staticmethod
    def _populate_masks(mask_factory: MaskFactory, array: Array, labels: NDArray[bool], rng: np.random.Generator):
        """
        Populate the array with masks generated by the mask factory.

        Parameters
        ----------
        mask_factory : MaskFactory
            Factory to generate masks.
        array : Array
            Array to populate with masks.
        labels : NDArray[bool]
            Binary labels for the samples.
        rng : np.random.Generator
            Random number generator for mask generation.

        Returns
        -------
        Array
            Array populated with masks.
        """
        for i in range(len(array)):
            array[i] = mask_factory.get_masks(labels, rng)
        return array

    @cached_property
    def masks(self) -> Array:
        """
        Masks array (shape: [n_models, n_samples]).
        """
        path = None if self.in_memory else os.path.join(self.path, f"batch_{self.batch_starter_seed}_masks.npy")
        populate_masks = True if self.in_memory or not os.path.exists(path) else False
        temporary_masks = self._create_array(self.in_memory, path,
            bool, (self.n_models, self.n_samples)
        )
        if populate_masks:
            return self._populate_masks(self.mask_factory, temporary_masks, self.labels, self.rng)
        else:
            return temporary_masks

    @cached_property
    def margins(self) -> Array:
        """
        Margins array (shape: [n_models, n_samples]).
        """
        return self._create_array(self.in_memory, None if self.in_memory else os.path.join(self.path, f"batch_{self.batch_starter_seed}_margins.npy"),
            np.float32, (self.n_models, self.n_samples)
        )

    @cached_property
    def test_accuracies(self) -> Array:
        """
        Test accuracies array (shape: [n_models]).
        """
        return self._create_array(self.in_memory, None if self.in_memory else os.path.join(self.path, f"batch_{self.batch_starter_seed}_test_accuracies.npy"), 
            np.float32, (self.n_models,)
        )
    
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
        return self.test_accuracies[instance_index] != 0

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
        self.margins[instance_index] = margins
        self.test_accuracies[instance_index] = test_accuracy
