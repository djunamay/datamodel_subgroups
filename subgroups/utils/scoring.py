from typing import Protocol
from numpy.typing import NDArray
import numpy as np
from ..models import SklearnClassifier

class MarginCalculator(Protocol):
    """
    A function that returns model margins for a given set of features and labels.
    """
    def __call__(self, model: SklearnClassifier, features: NDArray[float], labels: NDArray[bool]) -> NDArray[float]:
        ...

def compute_margins_from_probabilities(model: SklearnClassifier, features: NDArray[float], labels: NDArray[bool]) -> NDArray[float]:
    '''
    Returns model margins for a given set of features and labels.
    '''
    all_probabilities = model.predict_proba(features)[:,1]
    return (labels*2-1)*(np.log(all_probabilities)-np.log(1-all_probabilities))

class SignalNoiseRatioCalculator(Protocol):
    """
    A function that returns model margins for a given set of features and labels.
    """
    def __call__(self, margins: NDArray[float], masks: NDArray[bool]) -> NDArray[float]:
        ...

def compute_signal_noise(margins: NDArray[float], masks: NDArray[bool])-> NDArray[float]:
    """
    Compute the signal-to-noise ratio for a given set of margins and masks.
    Margins and masks must have shape (n_models, n_splits, n_samples).
    """
    masked_margins = np.ma.array(margins, mask=masks)
    signal = np.var(masked_margins.mean(axis=0), axis=0)
    noise = np.mean(np.var(masked_margins, axis=0), axis=0)
    return np.array(signal/noise)