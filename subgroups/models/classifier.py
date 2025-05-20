from .base import ModelFactory, SklearnClassifier, ModelFactoryInitializer
from xgboost import XGBClassifier
import chz
import numpy as np
from numpy.random import Generator

@chz.chz
class XgbFactory(ModelFactory):
    """
    Factory class for creating instances of XGBClassifier with specified hyperparameters.

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
    min_child_weight : int
        Minimum sum of instance weight (hessian) needed in a child.
    """
    learning_rate: float = chz.field(default=0.1, doc='Learning rate for the model')
    max_depth: int = chz.field(default=6, doc='Maximum depth of the tree')
    n_estimators: int = chz.field(default=100, doc='Number of trees in the model')
    reg_lambda: float = chz.field(default=1.0, doc='Regularization parameter for the model')
    reg_alpha: float = chz.field(default=0.0, doc='Regularization parameter for the model')
    subsample: float = chz.field(default=0.8, doc='Subsample ratio for the model')
    colsample_bytree: float = chz.field(default=0.8, doc='Column subsample ratio for the model')
    gamma: float = chz.field(default=0.0, doc='Minimum loss reduction required to make a split')
    min_child_weight: float = chz.field(default=1, doc='Minimum sum of instance weight (hessian) needed in a child')

    @staticmethod
    def _random_state(seed: int) -> np.random.RandomState:
        """
        Initialize a random state with a specified seed.

        Parameters
        ----------
        seed : int
            Seed for random number generation.

        Returns
        -------
        np.random.RandomState
            Initialized random state.
        """
        return np.random.default_rng(seed)
    
    def build_model(self, seed: int = None) -> SklearnClassifier:
        """
        Construct an XGBClassifier instance with the specified hyperparameters.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. Defaults to None.

        Returns
        -------
        SklearnClassifier
            An instance of XGBClassifier configured with the specified parameters.
        """
        return XGBClassifier(learning_rate=self.learning_rate, 
                             max_depth=self.max_depth, 
                             n_estimators=self.n_estimators, 
                             reg_lambda=self.reg_lambda, 
                             reg_alpha=self.reg_alpha, 
                             subsample=self.subsample, 
                             colsample_bytree=self.colsample_bytree, 
                             gamma=self.gamma, 
                             min_child_weight=self.min_child_weight,
                             random_state=self._random_state(seed))
    


@chz.chz
class XgbFactoryInitializer(ModelFactoryInitializer):

    @staticmethod
    def _rngs(seed: int) -> dict[str, Generator]:
        # one Generator per parameter so draws are independent yet reproducible
        seq = np.random.SeedSequence(seed)
        _rngs: dict[str, Generator] = {
            name: np.random.default_rng(s)   # sub-seeds
            for name, s in zip(
                ["lr", "depth", "n_est", "l2", "l1",
                 "subsample", "colsample", "gamma", "min_child"],
                seq.spawn(9)
            )
        }
        return _rngs
    
    @staticmethod
    def _learning_rate(_rngs: dict[str, Generator]) -> float:
        # log-uniform over [1e-2, 3e-1]
        return 10.0 ** _rngs["lr"].uniform(-2.0, -0.52)

    @staticmethod
    def _max_depth(_rngs: dict[str, Generator]) -> int:
        # integers 3 … 10.   >10 rarely helps and can explode RAM
        return int(_rngs["depth"].integers(3, 11))

    @staticmethod
    def _n_estimators(_rngs: dict[str, Generator]) -> int:
        # trees per boost round – log-uniform 64 … 1024
        return int(2 ** _rngs["n_est"].integers(6, 11))

    @staticmethod
    def _reg_lambda(_rngs: dict[str, Generator]) -> float:
        # L2 (ridge) – log-uniform 1e-3 … 10
        return 10.0 ** _rngs["l2"].uniform(-3.0, 1.0)

    @staticmethod
    def _reg_alpha(_rngs: dict[str, Generator]) -> float:
        # L1 (lasso) – log-uniform 1e-4 … 1
        return 10.0 ** _rngs["l1"].uniform(-4.0, 0.0)

    @staticmethod
    def _subsample(_rngs: dict[str, Generator]) -> float:
        # row sampling – uniform 0.5 … 1.0
        return float(_rngs["subsample"].uniform(0.5, 1.0))

    @staticmethod
    def _colsample_bytree(_rngs: dict[str, Generator]) -> float:
        # column sampling – uniform 0.5 … 1.0
        return float(_rngs["colsample"].uniform(0.5, 1.0))

    @staticmethod
    def _gamma(_rngs: dict[str, Generator]) -> float:
        # min split loss – uniform 0 … 5
        return float(_rngs["gamma"].uniform(0.0, 5.0))

    @staticmethod
    def _min_child_weight(_rngs: dict[str, Generator]) -> float:
        # log-uniform 0.1 … 10
        return 10.0 ** _rngs["min_child"].uniform(-1.0, 1.0)

    def build_model_factory(self, seed: int) -> ModelFactory:
        _rngs = self._rngs(seed)
        xgb_factory = XgbFactory(learning_rate=self._learning_rate(_rngs), 
                                 max_depth=self._max_depth(_rngs), 
                                 n_estimators=self._n_estimators(_rngs), 
                                 reg_lambda=self._reg_lambda(_rngs), 
                                 reg_alpha=self._reg_alpha(_rngs), 
                                 subsample=self._subsample(_rngs), 
                                 colsample_bytree=self._colsample_bytree(_rngs), 
                                 gamma=self._gamma(_rngs), 
                                 min_child_weight=self._min_child_weight(_rngs))
        return xgb_factory
