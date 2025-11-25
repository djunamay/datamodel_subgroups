import numpy as np
import chz
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Tuple, Iterator, Union, Any
from ..datastorage.experiment import Experiment
from ..datastorage.mask_margin import MaskMarginStorageInterface
from ..datastorage.counterfactuals import CounterfactualOutputs


class ReturnCounterfactualOutputInterface:
    """A callable class that takes a MaskMarginStorageInterface and returns CounterfactualOutputs, for which a score property must exist."""

    def __call__(self, training_output: MaskMarginStorageInterface, n_models: int, split: NDArray[bool]) -> CounterfactualOutputs:
        ...