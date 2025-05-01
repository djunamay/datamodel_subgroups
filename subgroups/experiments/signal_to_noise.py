from typing import List
from numpy.typing import NDArray
from tqdm import tqdm
import numpy as np
from ..utils.scoring import MarginCalculator, SignalNoiseRatioCalculator
from ..models import SklearnClassifier
from ..datasamplers.base import MaskGenerator


import chz 
from typing import Callable

def compute_architecture_snr(margin_calculator: MarginCalculator, 
                             snr_calculator: SignalNoiseRatioCalculator, 
                             mask_generator: MaskGenerator, 
                             class_indices_0: NDArray[int], 
                             class_indices_1: NDArray[int], 
                             alpha: float, 
                             num_samples: int, 
                             features: NDArray[float], 
                             labels: NDArray[bool], 
                             models: List[SklearnClassifier], 
                             n_train_splits: int, 
                             show_progress: bool)-> NDArray[float]:
    """
    Train many models (same architecture, different initializations) on `n_train_splits` random masks.
    Return the signal-to-noise ratio for each sample.
    (i.e. how much does the model's prediction (margin) for a given held-out sample vary across different masks vs different model initializations).

    Returns
    -------
    snr : ndarray  shape (n_samples)
    """
    margins = np.empty((len(models), n_train_splits, num_samples), dtype=float)
    masks = np.empty((len(models), n_train_splits, num_samples), dtype=bool)

    iterator = (
        tqdm(range(n_train_splits), desc="training splits", disable=not show_progress)
        if show_progress else range(n_train_splits)
    )
    for i in iterator:
        current_mask = mask_generator(class_indices_0, class_indices_1, alpha, i, num_samples)
        train_features = features[current_mask]
        train_labels = labels[current_mask]

        for j, model in enumerate(models):
            model.fit(train_features, train_labels)
            margins[j, i] = margin_calculator(model, features, labels)
            masks[j, i] = current_mask

    snr = snr_calculator(margins, masks)
    return snr