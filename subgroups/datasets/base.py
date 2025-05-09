import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
import chz

@chz.chz
class DatasetInterface(ABC):
    """
    Interface for datasets providing essential properties and methods for accessing dataset characteristics.
    Implementations should provide access to features, labels, and descriptive data, ensuring a consistent interface.
    """    
    @property 
    @abstractmethod
    def features(self) -> NDArray[float]:
        """Feature matrix (shape: [n_samples, n_features])."""
        ...
    
    @property
    @abstractmethod
    def coarse_labels(self) -> NDArray[bool]:
        """Binary labels for classification (shape: [n_samples])."""
        ...

    @property
    @abstractmethod
    def fine_labels(self) -> NDArray[bool]:
        """Fine-grained labels for clustering (shape: [n_samples])."""
        ...

    @property
    @abstractmethod
    def descriptive_data(self) -> np.recarray:
        """Descriptive data as a record array (shape: [n_samples, n_descriptive_features])."""
        ...

    @property
    @abstractmethod
    def class_indices(self) -> tuple[NDArray[int], NDArray[int]]:
        """Indices for class 1 and class 0 samples."""
        ...

    @property
    @abstractmethod
    def num_samples(self) -> int:
        """Total number of samples."""
        ...

    @property
    @abstractmethod
    def num_features(self) -> int:
        """Number of features."""
        ...


@chz.chz
class BaseDataset(DatasetInterface):
    """
    Base dataset class providing common properties for dataset handling.

    This class serves as a foundation for datasets, offering essential properties 
    to access dataset characteristics such as class indices, number of samples, 
    and number of features.
    """

    @property
    def class_indices(self) -> tuple[NDArray[int], NDArray[int]]:
        """
        Tuple containing indices for class 1 and class 0 samples.
        """
        return np.argwhere(self.coarse_labels).flatten(), np.argwhere(~self.coarse_labels).flatten()

    @property
    def num_samples(self) -> int:
        """
        Total number of samples in the dataset.
        """
        return len(self.features)

    @property
    def num_features(self) -> int:
        """
        Number of features in the dataset.
        """
        return len(self.features[0])