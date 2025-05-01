from abc import abstractmethod
from numpy.typing import NDArray
import chz
from tqdm import tqdm
import numpy as np
from .base import ModelFactory
from ..datasamplers.base import MaskGenerator
from ..utils.scoring import MarginCalculator

@chz.chz
class MaskMarginStorageInterface():
    @property
    @abstractmethod
    def masks(self)-> NDArray[bool]:
        ...
    
    @property
    @abstractmethod
    def margins(self)-> NDArray[float]:
        ...
    
    @property
    @abstractmethod
    def test_accuracies(self)-> NDArray[float]:
        ...
        
    @property
    @abstractmethod
    def model_trained(self)-> bool:
        ...
    
    @abstractmethod
    def populate_storage(self)-> None:
        ...

@chz.chz
class MaskMarginStorage(MaskMarginStorageInterface):
    model_factory: ModelFactory
    mask_generator: MaskGenerator
    margin_calculator: MarginCalculator
    alpha: float
    num_samples: int
    class_indices_0: NDArray[int]
    class_indices_1: NDArray[int]
    features: NDArray[float]
    labels: NDArray[bool]
    seed_start: int
    n_train_splits: int
    show_progress: bool

    @chz.init_property
    def masks(self)-> NDArray[bool]:
        return np.zeros((self.n_train_splits, self.num_samples), dtype=bool)

    @chz.init_property
    def margins(self)-> NDArray[float]:
        return np.zeros((self.n_train_splits, self.num_samples), dtype=float)
    
    @chz.init_property
    def test_accuracies(self)-> NDArray[float]:
        return np.zeros((self.n_train_splits,), dtype=float)
    
    @chz.init_property
    def model_trained(self)-> bool:
        return np.zeros((self.n_train_splits,), dtype=bool)
    
    def _train_eval_model(self, seed: int):
        model = self.model_factory.build_model()
        current_mask = self.mask_generator(self.class_indices_0, self.class_indices_1, self.alpha, seed, self.num_samples)
        model.fit(self.features[current_mask], self.labels[current_mask])
        test_accuracy = model.score(self.features[~current_mask], self.labels[~current_mask])
        current_margin = self.margin_calculator(model, self.features, self.labels)
        return test_accuracy, current_margin, current_mask
    
    def populate_storage(self)-> None:
        rng_train = np.arange(self.seed_start, self.seed_start + self.n_train_splits)
        iterator = (
                    tqdm(rng_train, desc="training splits", disable=not self.show_progress)
                    if self.show_progress else rng_train
                )
        for seed in iterator:
            output = self._train_eval_model(seed)
            self.test_accuracies[seed] = output[0]
            self.margins[seed] = output[1]
            self.masks[seed] = output[2]
            self.model_trained[seed] = True
        
        