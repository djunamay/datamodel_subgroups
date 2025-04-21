import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
import chz

@chz.chz
class DatasetInterface(ABC):
    """Abstract dataset: declare fields and required methods."""    
    @property 
    @abstractmethod
    def features(self)-> NDArray[float]:
        """Return features [n_samples, n_features]"""
        ...
    
    @property
    @abstractmethod
    def coarse_labels(self)-> NDArray[bool]:
        """Return labels [n_samples]"""
        ...

    @property
    @abstractmethod
    def fine_labels(self)-> NDArray[bool]:
        """Return labels [n_samples]"""
        ...

    @property
    @abstractmethod
    def descriptive_data(self)-> np.recarray:
        """Return descriptive data [n_samples, n_descriptive_features]"""
        ...

    @property
    @abstractmethod
    def class_indices(self)-> tuple[NDArray[int], NDArray[int]]:
        """Return train indices for class 1 and class 0"""
        ...


@chz.chz
class BaseDataset(DatasetInterface):
    """Base dataset: define common functions."""

    @property
    def class_indices(self)-> tuple[NDArray[int], NDArray[int]]:
        return np.argwhere(self.coarse_labels).flatten(), np.argwhere(~self.coarse_labels).flatten()
