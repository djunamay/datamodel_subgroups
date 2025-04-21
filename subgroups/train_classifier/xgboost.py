import numpy as np
import pandas as pd
import chz  
from numpy.typing import NDArray
from .classifier import BaseClassifier
from .base import DataloaderInterface
from xgboost import XGBClassifier
@chz.chz
class XGBoostClassifier(BaseClassifier):
    """XGBoost classifier"""
    params: dict=chz.field(doc="Parameters for the XGBoost classifier")
    dataloader: DataloaderInterface=chz.field
    model_seed: int=chz.field(doc="Seed for the model")
    n_jobs: int=chz.field(doc="Number of jobs for the model")

    @chz.init_property
    def _trained_classifier(self)-> None:
        classifier = XGBClassifier(**self.params, random_state=self.model_seed, n_jobs=self.n_jobs)
        classifier.fit(self.dataloader.dataset.features[self.dataloader.train_indices], self.dataloader.dataset.coarse_labels[self.dataloader.train_indices])
        return classifier
    
    @chz.init_property
    def predictions(self)-> None:
        return self._trained_classifier.predict_proba(self.dataloader.dataset.features)[:,1]


