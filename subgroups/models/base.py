import numpy as np
import chz
from sklearn.base import BaseEstimator, ClassifierMixin

class SklearnClassifier(ClassifierMixin, BaseEstimator):
    """Represents a classifier that combines BaseEstimator and ClassifierMixin functionalities."""

@chz.chz
class ModelFactory:
    """
    Factory class for creating instances of SklearnClassifier.
    """

    def build_model(self, seed: int = None) -> SklearnClassifier:
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


@chz.chz
class ModelFactoryInitializer:
    """
    Factory class for creating instances of ModelFactory.
    """

    def build_model_factory(self, seed: int) -> ModelFactory:
        ...
