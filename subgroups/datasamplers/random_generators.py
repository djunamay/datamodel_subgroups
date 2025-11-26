import numpy as np
from functools import cached_property
from ..utils.random import fork_rng

class RandomGeneratorSNR:
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
        children = fork_rng(np.random.default_rng(batch_starter_seed), 6)
        self._rngs_build_model_seed = children[0]
        self._rngs_get_mask_seed = children[1]
        self._rngs_train_data_shuffle_seed = children[2]
        self._rngs_model_factory_seed = children[3]
        self._rngs_mask_factory_seed = children[4]
        self._rngs_n_pcs_seed = children[5]
        self.batch_starter_seed = batch_starter_seed
        
    def _draw_seed_once(self) -> int:
        return self._rngs_get_mask_seed.integers(0, 2**32 - 1)
    
    @property
    def model_build_rng(self) -> np.random.Generator:
        return self._rngs_build_model_seed
    
    @cached_property
    def mask_rng(self) -> np.random.Generator:
        return self._rngs_get_mask_seed
    
    @property 
    def train_data_shuffle_rng(self) -> np.random.Generator:
        return self._rngs_train_data_shuffle_seed
    
    @property
    def model_factory_rng(self) -> np.random.Generator:
        return self._rngs_model_factory_seed
    
    @property
    def mask_factory_rng(self) -> np.random.Generator:
        return self._rngs_mask_factory_seed
    
    @property
    def n_pcs_rng(self) -> np.random.Generator:
        return self._rngs_n_pcs_seed


class RandomGeneratorTC:
    """
    Generates random seeds for various components of TC experiments using a batch starter seed.
    All seeds are newly generated at each call.

    Parameters
    ----------
    batch_starter_seed : int
        Initial seed for generating all subsequent seeds.
    """
    def __init__(self, batch_starter_seed: int):
        children = fork_rng(np.random.default_rng(batch_starter_seed), 3) # TODO: Don't need separate rngs here; only fork if the sequence in which will run (eg parallelizing) in an unknown order
        self._rngs_build_model_seed = children[0]
        self._rngs_get_mask_seed = children[1]
        self._rngs_train_data_shuffle_seed = children[2]
    
    @property
    def mask_rng(self) -> np.random.Generator:
        return self._rngs_get_mask_seed
        
    @property
    def model_build_rng(self) -> np.random.Generator:
        return self._rngs_build_model_seed
    
    @property 
    def train_data_shuffle_rng(self) -> np.random.Generator:
        return self._rngs_train_data_shuffle_seed
    