import numpy as np
import chz
from sklearn.base import BaseEstimator, ClassifierMixin

class SklearnClassifier(ClassifierMixin, BaseEstimator):
    """Any object that is BOTH BaseEstimator and ClassifierMixin."""

@chz.chz
class ModelFactory(): # only methods on interfaces
    seed: int = chz.field(default=None, doc='Seed to initialize the model factory. By default, the seed is not set so the sequence of models built by this factory is non-deterministic.')

    @chz.init_property
    def _random_state(self) -> np.random.RandomState:
        return np.random.default_rng(self.seed)

    def build_model(self) -> SklearnClassifier:
        ...


    