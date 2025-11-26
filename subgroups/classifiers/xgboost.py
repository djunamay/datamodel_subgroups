from sklearn.base import ClassifierMixin, BaseEstimator
from xgboost import XGBClassifier
import chz
import numpy as np
from numpy.random import Generator
from functools import partial
from typing import Callable

class SklearnClassifier(ClassifierMixin, BaseEstimator):
    """Represents a classifiers that combines BaseEstimator and ClassifierMixin functionalities."""

model_factory_fn = Callable[[np.random.Generator], SklearnClassifier]
"""
Callable for creating instances of SklearnClassifier.

Parameters
----------
rng : np.random.Generator
    Random number generator for reproducibility.
"""

def model_factory_xgboost(rng: np.random.Generator, learning_rate: float=0.1, max_depth: int=6, n_estimators: int=100, reg_lambda: float=1.0, reg_alpha: float=0.0, subsample: float=0.8, colsample_bytree: float=0.8, gamma: float=0.0, min_child_weight: float=1) -> SklearnClassifier:
    """
    Construct an XGBClassifier instance with the specified hyperparameters.

    Attributes
    ----------
    learning_rate : float
        Learning rate for the model.
    max_depth : int
        Maximum depth of the tree.
    n_estimators : int
        Number of trees in the model.
    reg_lambda : float
        L2 regularization parameter for the model.
    reg_alpha : float
        L1 regularization parameter for the model.
    subsample : float
        Subsample ratio for the model.
    colsample_bytree : float
        Column subsample ratio for the model.
    gamma : float
        Minimum loss reduction required to make a split.
    min_child_weight : float
        Minimum sum of instance weight (hessian) needed in a child.
    """
    return XGBClassifier(learning_rate=learning_rate,
                             max_depth=max_depth,
                             n_estimators=n_estimators,
                             reg_lambda=reg_lambda,
                             reg_alpha=reg_alpha,
                             subsample=subsample,
                             colsample_bytree=colsample_bytree,
                             gamma=gamma,
                             min_child_weight=min_child_weight,
                             random_state=rng,
                             base_score=0.5)


model_factory_init_fn = Callable[[np.random.Generator], model_factory_fn]
"""
Callable for creating a model_factory_fn Callable with a specific combination of parameters.

Parameters
----------
rng : np.random.Generator
    Random number generator for reproducibility.
"""

def model_factory_init_xgboost(rng: np.random.Generator):
    """
    Returns a model_factory_fn Callable with sampled max depth.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducibility.
    """
    return partial(model_factory_xgboost, max_depth=int(rng.integers(3, 11)))


