from .base import DataModelFactory
from sklearn.linear_model import LassoCV
from sklearn.linear_model._coordinate_descent import LinearModelCV
from .base import SklearnRegressor
import chz

class SklearnRegressorCV(SklearnRegressor, LinearModelCV):
    """Represents a regressor that combines BaseEstimator and RegressorMixin functionalities."""


@chz.chz
class LassoFactory(DataModelFactory):
    n_lambdas: int=chz.field(default=50, doc='Number of lambdas to use for the LassoCV.')
    cv_splits: int=chz.field(default=5, doc='Number of splits to use for the LassoCV.')
    
    def build_model(self, seed: int = None) -> SklearnRegressorCV:
        return LassoCV(cv=self.cv_splits, random_state=seed, n_jobs=1, n_alphas=self.n_lambdas)
