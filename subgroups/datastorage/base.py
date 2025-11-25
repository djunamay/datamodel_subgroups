from dataclasses import dataclass
from functools import cached_property
from typing import Union
import numpy as np
from abc import abstractmethod, ABC
from numpy.typing import NDArray
from .experiment import Experiment

Array = Union[np.ndarray, np.memmap]


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
