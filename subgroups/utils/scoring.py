from typing import Protocol
from numpy.typing import NDArray
import numpy as np
from ..models import SklearnClassifier

def compute_margins(probabilities: NDArray[float], labels: NDArray[bool]) -> NDArray[float]:
    r"""
    Calculate model margins for given probabilities and binary labels.

    The margin for each sample is computed as:

    .. math::

        \text{margin}_i = (2y_i - 1) \cdot \left( \log(p_i) - \log(1 - p_i) \right)

    where:
        - :math:`y_i \in \{0, 1\}` is the true binary label for sample :math:`i`,
        - :math:`p_i` is the predicted probability of the positive class for sample :math:`i`,
        - The term :math:`2y_i - 1` maps labels to :math:`\\{-1, +1\\}`,
        - The log-odds expression :math:`\log(p_i) - \log(1 - p_i)` is the logit of the prediction.

    Parameters
    ----------
    probabilities : NDArray[float]
        Predicted probabilities for the positive class 
        (shape: [n_samples,]).
    labels : NDArray[bool]
        Binary labels indicating the true class 
        (shape: [n_samples,]).

    Returns
    -------
    NDArray[float]
        Computed margins for each sample 
        (shape: [n_samples,]).
    """
    probabilities = np.clip(probabilities, 1e-4, 1 - 1e-4) # for trivial examples

    return (labels*2-1)*(np.log(probabilities)-np.log(1-probabilities))


def compute_signal_noise(margins: NDArray[float], masks: NDArray[bool], ddof: int = 1) -> NDArray[float]:
    r"""
    Compute the signal-to-noise ratio (SNR) for given margins and masks.

    The SNR is computed as follows:

    .. math::

        \text{SNR}_i = \frac{\mathrm{Var}_{\text{split}}\left(\mathbb{E}_{\text{init}}\left[\text{margins}_{s,i}\right]\right)}{\mathbb{E}_{\text{split}}\left[\mathrm{Var}_{\text{init}}\left(\text{margins}_{s,i}\right)\right]}

    where:
        - :math:`\text{margins}_{s,i}` is the margin of sample :math:`i` in training split :math:`s` across different model initializations.
        - :math:`\mathbb{E}_{\text{init}}` denotes averaging across model initializations.
        - :math:`\mathrm{Var}_{\text{split}}` denotes variance computed across training splits.
        - :math:`\mathrm{Var}_{\text{init}}` denotes variance computed across model initializations.
        - :math:`\mathbb{E}_{\text{split}}` denotes averaging across training splits.

    Parameters
    ----------
    margins : NDArray[float]
        Margins for each sample 
        (shape: [n_model_initializations, n_train_splits, n_samples]).
    masks : NDArray[bool]
        Boolean masks indicating selected samples 
        (shape: [n_model_initializations, n_train_splits, n_samples]).

    Returns
    -------
    NDArray[float]
        Signal-to-noise ratio for each held-out sample (shape: [n_samples]).
        Note that a margin is excluded from the SNR computation for a given sample if that sample was part of the training set used to generate the margin.
    """
    margins_masked = np.ma.array(margins, mask=masks)
    E_init = np.ma.mean(margins_masked, axis=0)
    V_init = np.ma.var(margins_masked, axis=0, ddof=ddof)
    signal = np.ma.var(E_init, axis=0, ddof=ddof) 
    noise = np.ma.mean(V_init, axis=0)
    snr = signal/np.ma.maximum(noise, 1e-12)
    return snr.filled(np.nan)


