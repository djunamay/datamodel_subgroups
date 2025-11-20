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
    def untransformed_features(self):
        return self._features

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

import numpy as np
import chz
from numpy.typing import NDArray
from sklearn import datasets

@chz.chz
class SyntheticData(BaseDataset):

    n_samples: int = chz.field(doc = 'number of samples to generate per class', default=500)
    random_state: int = chz.field(doc = 'random state', default=170)
    n_features: int = chz.field(doc = 'number of features to sample', default=2)
    centers_class_1: NDArray[float] = chz.field(doc = 'cluster centers for class 1 of size (n_centers, n_features)')
    centers_class_0: NDArray[float] = chz.field(doc = 'cluster centers for class 0 of size (n_centers, n_features)')

    @chz.init_property 
    def _synthetic_data(self):
        X_class_1, y_class_1 = datasets.make_blobs(n_samples=self.n_samples, random_state=self.random_state, n_features=self.n_features, centers = self.centers_class_1)
        X_class_0, y_class_0 = datasets.make_blobs(n_samples=self.n_samples, random_state=self.random_state, n_features=self.n_features, centers = self.centers_class_0)
        features = np.concatenate((X_class_0, X_class_1), axis=0)
        fine_labels = np.concatenate((y_class_0, y_class_1+1))
        coarse_labels = np.concatenate((np.zeros_like(y_class_0), np.ones_like(y_class_1)))
        return features, coarse_labels.astype(bool), fine_labels

    @property 
    def features(self) -> NDArray[float]:
        """Feature matrix (shape: [n_samples, n_features])."""
        return self._synthetic_data[0]
        
    
    @property
    def coarse_labels(self) -> NDArray[bool]:
        """Binary labels for classification (shape: [n_samples])."""
        return self._synthetic_data[1]

    @property
    def fine_labels(self) -> NDArray[bool]:
        """Fine-grained labels for clustering (shape: [n_samples])."""
        return self._synthetic_data[2]

    @property
    def descriptive_data(self) -> np.recarray:
        """Descriptive data as a record array (shape: [n_samples, n_descriptive_features])."""
        return None

    @property 
    def untransformed_features(self) -> NDArray[float]:
        """Feature matrix (shape: [n_samples, n_features])."""
        self._synthetic_data[0]