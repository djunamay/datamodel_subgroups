import numpy as np
import chz
from abc import ABC, abstractmethod
from numpy.typing import NDArray

class SplitResultsInterface(ABC):

    @abstractmethod
    def labels(self)-> np.ndarray[bool]:
        """
        Labels of the split.
        """
        ...

    @abstractmethod
    def probabilities_on_split(self)-> np.ndarray[float]:
        """
        Probabilities estimated by a model trained on training subset of this split.
        """
        ...

    @abstractmethod
    def probabilities_outside_split(self)-> np.ndarray[float]:
        """
        Probabilities estimated by a model trained on training subset not in this split.
        """
        ...
     

class CounterfactualResultsInterface(ABC):

    @abstractmethod
    def split_a(self)-> SplitResultsInterface:
        """
        Results for split A.
        """
        ...

    @abstractmethod
    def split_b(self)-> SplitResultsInterface:
        """
        Results for split B.
        """
        ...

@chz.chz
class CounterfactualEvaluationInterface(ABC):

    @abstractmethod
    def counterfactual_evaluation(self, cluster_labels)-> CounterfactualResultsInterface:
        ...

class PartitionStorageInterface(ABC):
    """
    Takes some input and returns a numpy array of integers, where each integer corresponds to a partition of the input.
    The length of the array is equal to the number of samples in the major class of interest.
    """
    @property
    @abstractmethod
    def partitions(self)-> NDArray[int]:
        ...

    @property
    @abstractmethod
    def n_partitions(self)-> int:
        ...

@chz.chz
@abstractmethod
class CounterfactualInputsInterface(ABC):
    @abstractmethod
    @chz.init_property
    def pca_input(self)->np.ndarray:
        ...
    
    @abstractmethod
    @chz.init_property
    def datamodel_input(self)->np.ndarray:
        ...

    @abstractmethod
    @chz.init_property
    def pca_filtered_input(self)->np.ndarray:
        ...
    