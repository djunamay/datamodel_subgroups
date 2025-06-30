from typing import Protocol
from numpy.typing import NDArray
import numpy as np
import chz
from abc import ABC, abstractmethod
from functools import cached_property

@chz.chz
class MaskFactory():
    """
    Interface for generating boolean masks to select training samples based on provided labels.

    Implementations of this class generate masks indicating the subset of samples to be used for training,
    ensuring flexibility in how subsets are chosen (e.g., balanced, random, or stratified).
    """

    def get_masks(self, labels: NDArray[bool], rng: np.random.Generator) -> NDArray[bool]:
        """
        Generate a boolean mask to select samples for training.

        Parameters
        ----------
        labels : NDArray[bool]
            Binary array indicating class membership of samples (`True` or `False`).
        rng : np.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        NDArray[bool]
            Boolean mask array, where `True` marks samples selected for training.
        """
        ...

@chz.chz
class MaskFactoryInitializer:

    def build_mask_factory(self, rng: np.random.Generator) -> MaskFactory:
        ...


class RandomGeneratorSNRInterface(ABC):
    """
    Every concrete class that implements this interface must

    • produce an *unchanging* mask_seed for the lifetime of the object
    • produce a *fresh* seed for the other methods on every call
    • Takes a batch_starter_seed as input, which is used to initialize the random number generator.

    Parameters
    ----------
    batch_starter_seed : int.
        The seed for the random number generator.
    """

    @abstractmethod
    def _draw_seed_once(self) -> int:
        """Generate a single seed that will be reused forever."""
        ...

    # ---------- fixed-per-instance seed ----------
    @cached_property           # <— cached on first access ☑
    def mask_seed(self) -> int:
        """
        Produce the *unchanging* seed for the mask factory (MaskFactory.get_masks(seed=...)) on every call.
        """
        # subclasses decide *how* it is drawn, but only once
        return self._draw_seed_once()
    
    # ---------- fresh-per-call seeds -------------
    @abstractmethod
    def model_build_rng(self) -> np.random.Generator: 
        """
        Produce a *fresh* seed for the model build method (ModelFactory.build_model(seed=...)) on every call.
        """
        ...

    @abstractmethod
    def train_data_shuffle_rng(self) -> np.random.Generator: 
        """
        Produce a *fresh* seed for the train data shuffle (train_one_classifier(shuffle_seed=...)) on every call.
        """
        ...

    @abstractmethod
    def model_factory_rng(self) -> np.random.Generator: 
        """
        Produce a *fresh* seed for the model factory (class ModelFactoryInitializer.build_model_factory(seed=...)) on every call.
        """
        ...

    @abstractmethod
    def mask_factory_rng(self) -> np.random.Generator: 
        """
        Produce a *fresh* seed for the mask factory (class MaskFactoryInitializer.build_mask_factory(seed=...)) on every call.
        """
        ...


class RandomGeneratorTCInterface(ABC):
    """
    Every concrete class that implements this interface must

    • produce a *fresh* seed for each method on every call
    • Takes a batch_starter_seed as input, which is used to initialize the random number generator.

    Parameters
    ----------
    batch_starter_seed : int.
        The seed for the random number generator.
    """

    @abstractmethod
    def model_build_rng(self) -> np.random.Generator:
        """
        Produce a *fresh* seed for the model build method (ModelFactory.build_model(seed=...)) on every call.
        """
        ...

    @abstractmethod
    def mask_rng(self) -> np.random.Generator:
        """
        Produce a *fresh* seed for the mask factory (MaskFactory.get_masks(seed=...)) on every call.
        """
        ...

    @abstractmethod
    def train_data_shuffle_rng(self) -> np.random.Generator:
        """
        Produce a *fresh* seed for the train data shuffle (train_one_classifier(shuffle_seed=...)) on every call.
        """
        ...

