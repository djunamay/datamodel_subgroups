import numpy as np
import chz
from sklearn.base import BaseEstimator, ClassifierMixin

class SklearnClassifier(ClassifierMixin, BaseEstimator):
    """Represents a classifier that combines BaseEstimator and ClassifierMixin functionalities."""

@chz.chz
class ModelFactory:
    """
    Factory class for creating instances of SklearnClassifier.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducibility.
    """

    def build_model(self, rng: np.random.Generator) -> SklearnClassifier:
        """
        Construct a SklearnClassifier instance.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator for reproducibility.

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

    def build_model_factory(self, rng: np.random.Generator) -> ModelFactory:
        ...
