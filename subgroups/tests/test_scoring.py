import numpy as np
import math
import pytest
from ..utils.scoring import compute_margins, compute_signal_noise

def test_compute_margins_basic():
    """
    Test compute_margins with a mix of True and False labels.

    Expected behavior:
      - For a probability of 0.1 with label False, the margin is:
        -(log(0.1) - log(0.9))
      - For a probability of 0.9 with label True, the margin is:
        log(0.9) - log(0.1)
      - For a probability of 0.5 with label False, the margin is 0.
    """
    probabilities = np.array([0.1, 0.9, 0.5])
    labels = np.array([False, True, False])
    expected = np.array([
        -(math.log(0.1) - math.log(0.9)),
         math.log(0.9) - math.log(0.1),
         0.0
    ])
    result = compute_margins(probabilities, labels)
    assert np.allclose(result, expected)

def test_compute_margins_shape_mismatch():
    """
    Test compute_margins with mismatched shapes to ensure it raises a ValueError.
    """
    probabilities = np.array([0.1, 0.9])
    labels = np.array([False, True, False])  # Mismatched length compared to probabilities
    with pytest.raises(ValueError):
        compute_margins(probabilities, labels)

def test_compute_signal_noise():
    """
    Test compute_signal_noise calculates correct signal-to-noise ratio given margins and masks.
    None of the margins are masked in this example.
    The test uses a small example with two training splits, two samples, and two model initiations.
    For sample 0, the computed SNR should be 4.5 and for sample 1, it should be 2.5.
    """
    margins = np.array([
        [[1.0, 1.0], [3.0, 2.0]],
        [[2.0, 3.0], [4.0, 6.0]]
    ])
    masks = np.zeros_like(margins, dtype=bool)
    expected = np.array([4.5, 2.5])
    result = compute_signal_noise(margins, masks)
    np.testing.assert_allclose(result, expected)

def test_compute_signal_noise_with_masks():
    """
    Test compute_signal_noise calculates correct signal-to-noise ratio given margins and masks.
    Margin 0,1,1 is masked in this example, meaning that the margin is not used in the computation of the SNR.
    The test uses a small example with two training splits, two samples, and two model initiations.
    For sample 0, the computed SNR should be 4.5 and for sample 1, it should be 2.
    """
    margins = np.array([
        [[1.0, 1.0], [3.0, 2.0]],
        [[2.0, 3.0], [4.0, 6.0]]
    ])
    masks = np.zeros_like(margins, dtype=bool)
    masks[0,1,1] = True
    expected = np.array([4.5, 2])
    result = compute_signal_noise(margins, masks)
    np.testing.assert_allclose(result, expected)
