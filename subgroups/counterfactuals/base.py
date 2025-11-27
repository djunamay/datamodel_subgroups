import numpy as np
import chz
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Tuple, Iterator, Union, Any
from ..storage.experiment import Experiment
from ..storage.training import BaseStorage
from ..storage.counterfactuals import CounterfactualOutputs


class ReturnCounterfactualOutputInterface:
    """A callable class that takes a MaskMarginStorageInterface and returns CounterfactualOutputs, for which a score property must exist."""

    def __call__(self, training_output: BaseStorage, n_models: int, split: NDArray[bool]) -> CounterfactualOutputs:
        ...