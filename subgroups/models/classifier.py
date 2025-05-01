from .base import ModelFactory, SklearnClassifier
from xgboost import XGBClassifier
import chz

@chz.chz
class XgbFactory(ModelFactory):

    def build_model(self) -> SklearnClassifier:
        return XGBClassifier(**self.params)