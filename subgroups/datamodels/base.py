from sklearn.base import BaseEstimator, RegressorMixin

import chz
from abc import ABC, abstractmethod

@chz.chz
class DatamodelsPipelineInterface(ABC):
    """
    Interface for datamodels pipeline.
    """
    @abstractmethod
    def fit_datamodels(self, indices, seed):
        ...


class SklearnRegressor(RegressorMixin, BaseEstimator):
    """Represents a regressor that combines BaseEstimator and RegressorMixin functionalities."""


@chz.chz
class DataModelFactory:
    """
    Factory class for creating instances of SklearnClassifier.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. Defaults to None.
    """

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