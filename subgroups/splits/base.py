import numpy as np
import chz
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Tuple, Iterator, Union, Any
from ..storage.experiment import Experiment
from ..storage.training import BaseStorage
from ..storage.counterfactuals import CounterfactualOutputs

from ..counterfactuals.base import ReturnCounterfactualOutputInterface
from ..pipelines.pipeline_cfl import pipeline_cfl

from numpy.typing import NDArray 
import numpy as np

import chz
from abc import ABC, abstractmethod

@chz.chz
class SplitterArgsInterface(ABC):
    ...


class SplitFactoryInterface(ABC): #TODO make this into a function
    """
    Class that takes as input a dataclass of type ResultsStorageInterface and returns a boolean split vector of length N_samples.
    """
    A : NDArray[float]
    r : NDArray[bool]
    
    @abstractmethod
    def split(self, k, SplitterArgsInterface) -> NDArray[bool]:
        ...

class ReturnCounterfactualOutputInterface:
    """A callable class that takes a MaskMarginStorageInterface and returns CounterfactualOutputs, for which a score property must exist."""

    def __call__(self, training_output: BaseStorage, n_models: int, split: NDArray[bool]) -> CounterfactualOutputs:
        ...


class ReturnBestSplitInterface(ABC):

    """
    Interface for class that implements method 'best_split', which takes as input an array of parameters over which to iterate and generate splits,
    then picks the best value based on some split score and returns a boolean NDArray of size (experiment.dataset.num_samples, ) indicating split for best split score.
    """

    @abstractmethod
    def best_split(self, K, n_models, batch_starter_seed, in_memory):
        """
        Parameters
        ----------
        K : NDArray[float]
            Array of parameters of values used by SplitFactoryInterface to generate splits.
        n_models : int
            Number of models to perform counterfactual pipelines on.
        batch_starter_seed : int
            Initial seed used for reproducible batching.
        in_memory : bool
            If True, run computation without writing intermediate results to disk.
        
        Returns
        -------
        NDArray[bool]
            Array of shape (experiment.dataset.num_samples,) indicating best splits.
        """
        ...

    @staticmethod
    def get_true_scores_for_splits(K: NDArray[float], 
                      experiment: Experiment, 
                      n_models: int, 
                      batch_starter_seed: int, 
                      in_memory: bool, 
                      splitter: SplitFactoryInterface, 
                      return_counterfactual_outputs: ReturnCounterfactualOutputInterface,
                      SplitArgs: SplitterArgsInterface):

        """
        Returns counterfactual training scores from ReturnCounterfactualOutputInterface class, as computed for data splits definied in SplitFactoryInterface
        across an array of parameters of values used by SplitFactoryInterface to generate splits.

        Parameters
        ----------
        K : NDArray[float]
            Array of parameters of values used by SplitFactoryInterface to generate splits.
        experiment : Experiment
            Experiment object containing dataset, metadata, and storage utilities.
        n_models : int
            Number of models to perform counterfactual pipelines on.
        batch_starter_seed : int
            Initial seed used for reproducible batching.
        in_memory : bool
            If True, run computation without writing intermediate results to disk.
        splitter : SplitFactoryInterface
            Splitter object defining how to construct the binary split.
        return_counterfactual_outputs : ReturnCounterfactualOutputInterface
            Utility class that generates counterfactual outputs for each model. Must contain a ".score" attribute.

        Returns
        -------
        NDArray[float]
            Array of shape (len(K),) containing counterfactual outputs for each model.

        """
        subtype_scores = np.empty((len(K), n_models)) 

        for i,k in enumerate(K):
            train_out = pipeline_cfl(
                            experiment=experiment,
                            split=splitter.split(k, SplitArgs), # an instance should not start with a capital; user should use a partial function instead so don't need all of this, it can just be a function
                            n_models=n_models,
                            batch_starter_seed=batch_starter_seed,
                            in_memory=in_memory,
                            return_counterfactual_outputs=return_counterfactual_outputs, 
                        )
            subtype_scores[i] = train_out.score

        return subtype_scores

class ProcessExperimentForSplitsInterface:
    """A callable class that takes an experiment of class Experiment and returns an NDArray used as input to SplitFactoryInterface"""

    def __call__(self, experiment) -> NDArray:
        ...