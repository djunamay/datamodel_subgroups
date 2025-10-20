from dataclasses import dataclass
from functools import cached_property
from typing import Union
import numpy as np
from abc import abstractmethod, ABC
from .experiment import Experiment
from numpy.typing import NDArray


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

class CombinedMaskMarginStorageInterface(ABC):

    @property
    def masks(self) -> Array:
        ...
    
    @property
    def margins(self) -> Array:
        ...
        

@dataclass
class ResultsStorageInterface(ABC):
    """
    Interface for returning results from a datamodel fitting experiment.
    Expected filename suffixes (per batch):
      - 'weights', 'pearson_correlations', 'spearman_correlations', 'rmse', 'biases'
    """ 

    experiment: Experiment

    @abstractmethod 
    @cached_property
    def weights(self) -> NDArray: 
        """datamodel experiment weights (shape: [n_samples, n_samples])""" 

    @abstractmethod 
    @cached_property
    def pearson(self) -> NDArray: 
        """datamodel experiment pearson correlations (shape: [n_samples])""" 

    @abstractmethod 
    @cached_property
    def spearman(self) -> NDArray: 
        """datamodel experiment spearman correlations (shape: [n_samples])""" 
    
    @abstractmethod 
    @cached_property
    def rmse(self) -> NDArray: 
        """datamodel experiment RMSE values (shape: [n_samples])""" 
    
    @cached_property
    @abstractmethod 
    def bias(self) -> NDArray: 
        """datamodel experiment bias values (shape: [n_samples])""" 

    @abstractmethod
    @cached_property
    def sample_indices(self) -> NDArray:
        """datamodel experiment sample indices corresponding to the order of rows in the weights matrix"""
