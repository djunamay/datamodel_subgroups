from sklearn.base import BaseEstimator, RegressorMixin
from typing import List, Dict, Any, Union
import chz
from abc import ABC, abstractmethod


class DatamodelsPipelineInterface(ABC):
    """
    Interface for datamodels pipeline.
    Datamodels pipelines should be implemented as a class that implements this interface to allow for maximum flexibility in how the datamodels are fit.
    E.g. the user may want to use the Fast_l1 solver (https://github.com/MadryLab/fast_l1) if they have access to a GPU.
    The only requirement for the fit_datamodels method is that it takes the following input parameters:

    Parameters
    ----------
    seed : int
        Random seed for reproducibility. Defaults to None. Allows the pipeline to optionally interface with batch IDs if jobs are run in parallel.
    n_train : int
        Number of training mask-margin pairs to use for training the datamodels.
    n_test : int
        Number of test mask-margin pairs to use for testing the datamodels. Defaults to None, in which case all remaining mask-margin pairs are used for testing.
    indices : List[int]
        List of indices specifying the held-out samples for which the datamodels should be fit.
    in_memory : bool
        Whether to fit the datamodels in memory or to save them to disk.

    Returns
    -------
    Union[Dict[str, Any], str]
        - If `in_memory=True`: returns a dictionary with model outputs.
        - If `in_memory=False`: returns a string path to the output location on disk.
    """
    @abstractmethod
    def fit_datamodels(self, indices: List[int], n_train: int, n_test: int=None, seed: int=None, in_memory: bool=True) -> Union[Dict[str, Any], str]:
        ...


class SklearnRegressor(RegressorMixin, BaseEstimator):
    """Represents a regressor that combines BaseEstimator and RegressorMixin functionalities."""


@chz.chz
class DataModelFactory(ABC):
    """
    Factory class for creating instances of SklearnClassifier.
    Allows users to specify a different model type in the form of an SklearnRegressor as part of the DatamodelsPipelineBasic class.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. Defaults to None.
    """

    @abstractmethod
    def build_model(self, seed: int = None) -> SklearnRegressor:
        """
        Construct a SklearnClassifier instance.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. Defaults to None.

        Returns
        -------
        SklearnClassifier
            An instance of SklearnClassifier.
        """
        ...