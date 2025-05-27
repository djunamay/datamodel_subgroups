from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from .base import DatasetInterface
import chz

@chz.chz
class RandomDataset(DatasetInterface):
    """
    Fast, purely in-memory implementation of `DatasetInterface` that
    generates reproducible, random data for tests and benchmarks.
    """
    n_samples: int = 1_000
    n_features: int = 20
    n_descriptive_features: int = 3
    seed: int | None = None
    rng = np.random.default_rng(seed)

    @chz.init_property
    def _features(self):
        return self.rng.random((self.n_samples, self.n_features), dtype=float)

    @chz.init_property
    def _coarse_labels(self):
        return self.rng.random(self.n_samples) > 0.5

    @chz.init_property
    def _fine_labels(self):
        empty = np.zeros(self.n_samples)
        empty[self.coarse_labels] = self.rng.integers(0, 3, np.sum(self.coarse_labels))
        empty[~self.coarse_labels] = self.rng.integers(3, 6, np.sum(~self.coarse_labels))
        return empty

    @chz.init_property
    def _descriptive_data(self):
        desc_raw = self.rng.random((self.n_samples, self.n_descriptive_features))
        return np.core.records.fromarrays(
            desc_raw.T,
            names=[f"d{i}" for i in range(self.n_descriptive_features)],
        )

    @chz.init_property
    def _class1_idx(self):
        return np.flatnonzero(self._coarse_labels)

    @chz.init_property
    def _class0_idx(self):
        return np.flatnonzero(~self._coarse_labels)

    # ---------------------------- Interface ---------------------------------
    @property
    def features(self) -> NDArray[float]:
        return self._features

    @property
    def coarse_labels(self) -> NDArray[bool]:
        return self._coarse_labels

    @property
    def fine_labels(self) -> NDArray[bool]:
        return self._fine_labels

    @property
    def descriptive_data(self) -> np.recarray:
        return self._descriptive_data

    @property
    def class_indices(self) -> Tuple[NDArray[int], NDArray[int]]:
        return self._class1_idx, self._class0_idx

    @property
    def num_samples(self) -> int:
        return self._features.shape[0]

    @property
    def num_features(self) -> int:
        return self._features.shape[1]