from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.linear_model import LassoCV
from sklearn.linear_model._coordinate_descent import LinearModelCV
import chz
from sklearn.linear_model import LinearRegression
from typing import Callable


class SklearnRegressor(RegressorMixin, BaseEstimator):
    """Represents a regressor that combines BaseEstimator and RegressorMixin functionalities."""


class SklearnRegressorCV(SklearnRegressor, LinearModelCV):
    """A class that combines the functionalities of SklearnRegressor and LinearModelCV."""

datamodel_factory_fn = Callable[[int], SklearnRegressorCV | SklearnRegressor]
"""
Callable that returns instance of SklearnRegressorCV.

Parameters
----------
seed : int, optional
    Random seed.
"""

def datamodel_factory_lasso(seed: int, n_lambdas: int=50, cv_splits: int=5) -> datamodel_factory_fn:
    """
    A factory for creating LassoCV models.

    Parameters
    ----------
    n_lambdas : int, optional
        Number of lambdas to use for the LassoCV. Defaults to 50.
    cv_splits : int, optional
        Number of splits to use for the LassoCV. Defaults to 5.
    """
    return LassoCV(cv=cv_splits, random_state=seed, n_jobs=1, n_alphas=n_lambdas)

def datamodel_factory_linear(seed: int) -> datamodel_factory_fn:
    """
    A factory for creating unregularized LinearRegression models.
    """
    return LinearRegression()
