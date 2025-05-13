import chz
from abc import ABC, abstractmethod
from numpy.random import Generator
import numpy as np
from .base import RandomGeneratorSNRInterface, RandomGeneratorTCInterface
from functools import cached_property

class RandomGeneratorSNR(RandomGeneratorSNRInterface):
    """
    Generates independent random seeds for various components of the SNR experiment using a batch starter seed.

    This class provides seeds for model building, mask generation, data shuffling, model factory, and mask factory.
    The mask seed remains constant across calls, while other seeds are newly generated each time.

    Parameters
    ----------
    batch_starter_seed : int
        Initial seed for generating all subsequent seeds.
    """

    def __init__(self, batch_starter_seed: int):
        self.batch_starter_seed = batch_starter_seed
        self._seq = np.random.SeedSequence(self.batch_starter_seed)
        self._rngs_build_model_seed = np.random.default_rng(self._seq.spawn(1)[0])
        self._rngs_get_mask_seed = np.random.default_rng(self._seq.spawn(1)[0])
        self._rngs_train_data_shuffle_seed = np.random.default_rng(self._seq.spawn(1)[0])
        self._rngs_model_factory_seed = np.random.default_rng(self._seq.spawn(1)[0])
        self._rngs_mask_factory_seed = np.random.default_rng(self._seq.spawn(1)[0])
        
    def _draw_seed_once(self) -> int:
        return self._rngs_get_mask_seed.integers(0, 2**32 - 1)
    
    @property
    def model_build_seed(self) -> int:
        return self._rngs_build_model_seed.integers(0, 2**32 - 1)
    
    @cached_property
    def mask_seed(self) -> int:
        return self._draw_seed_once()
    
    @property 
    def train_data_shuffle_seed(self) -> int:
        return self._rngs_train_data_shuffle_seed.integers(0, 2**32 - 1)
    
    @property
    def model_factory_seed(self) -> int:
        return self._rngs_model_factory_seed.integers(0, 2**32 - 1)
    
    @property
    def mask_factory_seed(self) -> int:
        return self._rngs_mask_factory_seed.integers(0, 2**32 - 1)


class RandomGeneratorTC(RandomGeneratorTCInterface):
    """
    Generates random seeds for various components of TC experiments using a batch starter seed.
    All seeds are newly generated at each call.

    Parameters
    ----------
    batch_starter_seed : int
        Initial seed for generating all subsequent seeds.
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
    