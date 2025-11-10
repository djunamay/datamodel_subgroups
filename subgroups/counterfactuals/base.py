import numpy as np
import chz
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Tuple, Iterator, Union, Any
from ..datastorage.experiment import Experiment
from ..datastorage.base import MaskMarginStorageInterface
from ..datastorage.counterfactuals import CounterfactualOutputs

class SplitFactoryInterface(ABC):
    """
    Class that takes as input a dataclass of type ResultsStorageInterface and returns a boolean split vector of length N_samples.
    """
    experiment: Experiment

    @property
    @abstractmethod
    def split(self) -> NDArray[bool]:
        ...

class ReturnCounterfactualOutputInterface:
    """A callable class that takes a MaskMarginStorageInterface and returns CounterfactualOutputs."""

    def __call__(self, training_output: MaskMarginStorageInterface, n_models: int, split: NDArray[bool]) -> CounterfactualOutputs:
        ...