import chz
from abc import ABC, abstractmethod
from numpy.random import Generator
import numpy as np



class RandomGeneratorSNRInterface(ABC):
    batch_starter_seed: int

    @abstractmethod
    def model_build_seed(self) -> int:
        ...

    @abstractmethod
    def mask_seed(self) -> int:
        ...

    @abstractmethod
    def train_data_shuffle_seed(self) -> int:
        ...
    
    @abstractmethod
    def model_factory_seed(self) -> int:
        ...
    
    @abstractmethod
    def mask_factory_seed(self) -> int:
        ...
    
class RandomGeneratorSNR(RandomGeneratorSNRInterface):
    """
    Random generator for SNR experiments.
    """
    def __init__(self, batch_starter_seed: int):
        self.batch_starter_seed = batch_starter_seed
        self._seq = np.random.SeedSequence(self.batch_starter_seed)
        self._rngs_build_model_seed = np.random.default_rng(self._seq.spawn(1)[0])
        self._rngs_get_mask_seed = np.random.default_rng(self._seq.spawn(1)[0])
        self._rngs_train_data_shuffle_seed = np.random.default_rng(self._seq.spawn(1)[0])
        self._rngs_model_factory_seed = np.random.default_rng(self._seq.spawn(1)[0])
        self._rngs_mask_factory_seed = np.random.default_rng(self._seq.spawn(1)[0])
        self._mask_seed = self._rngs_get_mask_seed.integers(0, 2**32 - 1)
        
    @property
    def model_build_seed(self) -> int:
        return self._rngs_build_model_seed.integers(0, 2**32 - 1)
    
    @property
    def mask_seed(self) -> int:
        return self._mask_seed
    
    @property 
    def train_data_shuffle_seed(self) -> int:
        return self._rngs_train_data_shuffle_seed.integers(0, 2**32 - 1)
    
    @property
    def model_factory_seed(self) -> int:
        return self._rngs_model_factory_seed.integers(0, 2**32 - 1)
    
    @property
    def mask_factory_seed(self) -> int:
        return self._rngs_mask_factory_seed.integers(0, 2**32 - 1)


class RandomGeneratorTCInterface(ABC):
    batch_starter_seed: int

    @abstractmethod
    def model_build_seed(self) -> int:
        ...

    @abstractmethod
    def mask_seed(self) -> int:
        ...

    @abstractmethod
    def train_data_shuffle_seed(self) -> int:
        ...



class RandomGeneratorTC(RandomGeneratorTCInterface):
    """
    Random generator for TC experiments.
    """
    def __init__(self, batch_starter_seed: int):
        self.batch_starter_seed = batch_starter_seed
        self._seq = np.random.SeedSequence(self.batch_starter_seed)
        self._rngs_build_model_seed = np.random.default_rng(self._seq.spawn(1)[0])
        self._rngs_get_mask_seed = np.random.default_rng(self._seq.spawn(1)[0])
        self._rngs_train_data_shuffle_seed = np.random.default_rng(self._seq.spawn(1)[0])
    
    @property
    def mask_seed(self) -> int:
        return self._rngs_get_mask_seed.integers(0, 2**32 - 1)
        
    @property
    def model_build_seed(self) -> int:
        return self._rngs_build_model_seed.integers(0, 2**32 - 1)
    
    @property 
    def train_data_shuffle_seed(self) -> int:
        return self._rngs_train_data_shuffle_seed.integers(0, 2**32 - 1)
    