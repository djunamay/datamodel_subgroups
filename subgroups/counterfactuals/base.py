import numpy as np
import chz
from abc import ABC, abstractmethod

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