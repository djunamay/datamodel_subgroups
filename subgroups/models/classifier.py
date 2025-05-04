from .base import ModelFactory, SklearnClassifier
from xgboost import XGBClassifier
import chz

@chz.chz
class XgbFactory(ModelFactory):
    learning_rate: float = chz.field(default=0.1, doc='Learning rate for the model')
    max_depth: int = chz.field(default=6, doc='Maximum depth of the tree')
    n_estimators: int = chz.field(default=100, doc='Number of trees in the model')
    reg_lambda: float = chz.field(default=1.0, doc='Regularization parameter for the model')
    reg_alpha: float = chz.field(default=0.0, doc='Regularization parameter for the model')
    subsample: float = chz.field(default=0.8, doc='Subsample ratio for the model')
    colsample_bytree: float = chz.field(default=0.8, doc='Column subsample ratio for the model')
    gamma: float = chz.field(default=0.0, doc='Minimum loss reduction required to make a split')
    min_child_weight: int = chz.field(default=1, doc='Minimum sum of instance weight (hessian) needed in a child')

    def build_model(self) -> SklearnClassifier:
        return XGBClassifier(learning_rate=self.learning_rate, 
                             max_depth=self.max_depth, 
                             n_estimators=self.n_estimators, 
                             reg_lambda=self.reg_lambda, 
                             reg_alpha=self.reg_alpha, 
                             subsample=self.subsample, 
                             colsample_bytree=self.colsample_bytree, 
                             gamma=self.gamma, 
                             min_child_weight=self.min_child_weight,
                             random_state=self._random_state)