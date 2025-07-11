import numpy as np
import chz
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Tuple, Iterator, Union, Any

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


@dataclass
class CounterfactualInputs:
    names: Tuple[str, ...] = field(init=False, repr=False)
    matrices: Tuple[np.ndarray, ...] = field(init=False, repr=False)

    def __init__(self, **named_matrices: Union[np.ndarray, Any]) -> None:
        """
        Accept any number of keyword‐arguments, where each key is a name
        and each value is a numpy array (or array‐like). E.g.:
        
            CounterfactualInputs(A=A, B=B, C=C)
        """
        # store the names in insertion order
        self.names = tuple(named_matrices.keys())
        # convert to numpy arrays
        self.matrices = tuple(np.asarray(m) for m in named_matrices.values())

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        """
        Iterate over (name, matrix) pairs.
        """
        return iter(zip(self.names, self.matrices))

    def __len__(self) -> int:
        """
        Number of matrices.
        """
        return len(self.matrices)

    def __getitem__(self, idx: Union[int, slice]
                    ) -> Union[Tuple[str, np.ndarray],
                               Tuple[Tuple[str, np.ndarray], ...]]:
        """
        Indexing or slicing returns name/matrix pairs.
        """
        items = tuple(zip(self.names, self.matrices))
        return items[idx]
    
class CounterfactualInputsInterface(ABC):
    """
    Abstract interface that defines the contract for building counterfactual inputs.
    All implementations must return a CounterfactualInputs instance.
    """
    @property
    @abstractmethod
    def to_counterfactual_inputs(self) -> CounterfactualInputs:
        """
        Build and return a CounterfactualInputs object containing all named input matrices.

        Returns:
            CounterfactualInputs: a data container of named matrices for counterfactual analysis.
        """
        ...