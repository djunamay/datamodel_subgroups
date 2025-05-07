from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Union, Optional
import numpy as np
from abc import abstractmethod
from subgroups.datasamplers.base import MaskFactory
from numpy.typing import NDArray
import warnings
Array = Union[np.ndarray, np.memmap]

@dataclass
class MaskMarginStorageInterface:
    """
    Interface for mask margin storage.
    """
    n_models: int
    n_samples: int
    labels: NDArray[bool]
    mask_factory: MaskFactory
    in_memory: bool = True
    path: Optional[Path] = None 

    @property
    @abstractmethod
    def masks(self) -> Array:
        """ Return masks [n_models, n_samples]"""
        ...
    
    @property
    @abstractmethod
    def margins(self) -> Array:
        """ Return margins [n_models, n_samples]"""
        ...

    @property
    @abstractmethod
    def test_accuracies(self) -> Array:
        """ Return test accuracies [n_models]"""
        ...
    
    @abstractmethod
    def is_filled(self, instance_index: int) -> bool:
        """ Return True if the instance is filled"""
        ...

    @abstractmethod
    def fill_results(self, instance_index: int, margins: Array, test_accuracy: float):
        """ Fill the results for the instance"""
        ...

@dataclass
class MaskMarginStorage(MaskMarginStorageInterface):
    n_models: int
    n_samples: int
    labels: NDArray[bool]
    mask_factory: MaskFactory
    in_memory: bool = True
    path: Optional[Path] = None   

    def __post_init__(self):
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
    def _create_array(in_memory: bool, path: Optional[Path], dtype: np.dtype, shape: tuple[int, int]) -> Array:
        if in_memory:
            return np.zeros(shape, dtype=dtype)

        if path is None:
            raise ValueError("path must be provided when in_memory=False")

        mode = "r+" if path.exists() else "w+"
        return np.memmap(path, dtype=dtype, mode=mode, shape=shape)

    @staticmethod
    def _populate_masks(mask_factory: MaskFactory, array: Array, labels: NDArray[bool]):
        for i in range(len(array)):
            array[i] = mask_factory.get_masks(labels)
        return array

    @cached_property
    def masks(self) -> Array:
        temporary_masks = self._create_array(self.in_memory, None if self.in_memory else self.path.with_suffix("_masks.npy"),
            bool, (self.n_models, self.n_samples)
        )
        return self._populate_masks(self.mask_factory, temporary_masks, self.labels)

    @cached_property
    def margins(self) -> Array:
        return self._create_array(self.in_memory, None if self.in_memory else self.path.with_suffix("_margins.npy"),
            np.float32, (self.n_models, self.n_samples)
        )

    @cached_property
    def test_accuracies(self) -> Array:
        return self._create_array(self.in_memory, None if self.in_memory else self.path.with_suffix("_test_accuracies.npy"), 
            np.float32, (self.n_models)
        )
    
    def is_filled(self, instance_index: int) -> bool:
        return self.test_accuracies[instance_index] != 0

    def fill_results(self, instance_index: int, margins: Array, test_accuracy: float):
        self.margins[instance_index] = margins
        self.test_accuracies[instance_index] = test_accuracy
