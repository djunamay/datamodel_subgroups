import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
import chz
from subgroups.datasets.base import DatasetInterface

@chz.chz
class DataloaderInterface(ABC):

    @property 
    @abstractmethod
    def train_indices(self)-> NDArray[int]:
        """Return train indices of length int(n_samples*alpha)"""
        ...
    
    @property
    @abstractmethod
    def test_indices(self)-> NDArray[int]:
        """Return test indices of length int(n_samples*(1-alpha))"""
        ...

@chz.chz
class DataLoader(DataloaderInterface):
    dataset: DatasetInterface=chz.field 
    alpha: float=chz.field(doc="Alpha for the dataloader")
    train_seed: int=chz.field(doc="Seed for the train indices")

    @chz.init_property
    def _rng(self)-> int:
        # create a small RNG seeded by the trial index
        rng = np.random.default_rng(self.train_seed)
        return rng

    @chz.init_property
    def train_indices(self)-> NDArray[int]:
        samples_per_class = int((len(self.dataset.features)*self.alpha)/2)
        indices_class_1 = self._rng.permutation(self.dataset.class_indices[0])
        indices_class_1_selected = indices_class_1[:samples_per_class]
        indices_class_0 = self._rng.permutation(self.dataset.class_indices[1])
        indices_class_0_selected = indices_class_0[:samples_per_class]
        train_indices = np.concatenate([indices_class_1_selected, indices_class_0_selected])
        return train_indices

    @chz.init_property
    def test_indices(self)-> NDArray[int]:
        return np.setdiff1d(np.arange(len(self.dataset.features)), self.train_indices)
    