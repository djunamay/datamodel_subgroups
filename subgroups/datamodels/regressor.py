from .base import DataModelFactory
from sklearn.linear_model import LassoCV
from sklearn.linear_model._coordinate_descent import LinearModelCV
from .base import SklearnRegressor
import chz
from sklearn.linear_model import LinearRegression

class SklearnRegressorCV(SklearnRegressor, LinearModelCV):
    """A class that combines the functionalities of SklearnRegressor and LinearModelCV."""


@chz.chz
class LassoFactory(DataModelFactory):
    """
    A factory for creating LassoCV models.

    Parameters
    ----------
    n_lambdas : int, optional
        Number of lambdas to use for the LassoCV. Defaults to 50.
    cv_splits : int, optional
        Number of splits to use for the LassoCV. Defaults to 5.
    """
    n_lambdas: int=chz.field(default=50, doc='Number of lambdas to use for the LassoCV.')
    cv_splits: int=chz.field(default=5, doc='Number of splits to use for the LassoCV.')
    
    def build_model(self, seed: int = None) -> SklearnRegressorCV:
        """
        Construct an SklearnRegressorCV instance based on an optional input seed.
        """
        return LassoCV(cv=self.cv_splits, random_state=seed, n_jobs=1, n_alphas=self.n_lambdas)

@chz.chz
class LinearRegressionFactory(DataModelFactory):
    """
    A factory for creating unregularized LinearRegression models.
    """

    def build_model(self, seed: int = None) -> SklearnRegressor:
        """
        Construct a LinearRegression instance (no regularization).
        """
        return LinearRegression()
