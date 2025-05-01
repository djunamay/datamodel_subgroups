import numpy as np
import chz
from sklearn.base import BaseEstimator, ClassifierMixin

class SklearnClassifier(ClassifierMixin, BaseEstimator):
    """Any object that is BOTH BaseEstimator and ClassifierMixin."""

@chz.chz
class ModelFactory():
    params: dict

    def build_model(self) -> SklearnClassifier:
        ...