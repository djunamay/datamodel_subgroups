
from ..datasamplers.mask_generators import CounterfactualMaskFactory
import numpy as np
from ..datastorage.registry import RandomDataset
from ..counterfactuals.outputs import ReturnCounterfactualOutputsBasic
import pytest

class _DummyStorage:
    """Mimics MaskMarginStorageInterface: holds margins, masks, labels."""
    def __init__(self, margins, masks, labels):
        self.margins = margins          # shape (n_models, n_samples)
        self.masks = masks              # shape (n_models, n_samples), True = used in training (exclude)
        self.labels = labels.astype(bool)  # shape (n_samples,)


def test_CounterfactualMaskFactory():
    """
    Test that the CounterfactualMaskFactory raises split valueerror as expected.
    """
    randomdata = RandomDataset()
    labels = randomdata.coarse_labels
    split = np.zeros(len(labels), dtype=bool)
    split[np.random.permutation(labels)] = np.random.randint(0, 2, np.sum(labels))
    counterfactual = CounterfactualMaskFactory(split=split, alpha=0.05)

    with pytest.raises(ValueError, match="Bool split vector must index samples from one class only."):
        counterfactual.get_masks(labels, np.random.default_rng(2))

    split = np.zeros(len(labels), dtype=bool)
    split[(labels)] = np.random.randint(0, 2, np.sum(labels))
    counterfactual = CounterfactualMaskFactory(split=split, alpha=0.05)
    assert len(counterfactual.get_masks(labels, np.random.default_rng(2))) == len(labels)


def test_ReturnCounterfactualOutputsBasic_return_accuracies():
    """
    Test accuracy calculation of ReturnCounterfactualOutputsBasic class with simple example.
    """
    func = ReturnCounterfactualOutputsBasic()

    n_models = 2
    n_samples = 3

    np.random.seed(42)

    labels = np.random.randint(0,2,(n_samples,))
    mask = np.random.randint(0,2,(n_models,n_samples))
    mask = mask.astype(bool)
    logits = np.random.randn(n_models,n_samples)

    expected_accuracies = np.array([1/2, 1])

    assert np.array_equal(expected_accuracies, func._return_accuracies(labels, logits, mask))


def test_ReturnCounterfactualOutputsBasic_Happy():

    func = ReturnCounterfactualOutputsBasic()

    labels = np.array([0,1,1,1, 0,1,0,1], dtype=bool)

    # split indexes a subset of class 1 only
    split = np.array([0,1,1,1, 0,0,0,0], dtype=bool)

    # Masks: Split A must have at least some True (used in training) for ALL models;
    # Split B must have NO True anywhere.
    # We'll set: model0 saw two split_A samples, model1 saw one split_A; none saw split_B.
    masks = np.zeros((2, labels.size), dtype=bool)
    masks[0, split] = np.array([True, True, False])  # first two True among split_A positions
    masks[1, np.where(split)[0][0]] = True

    # create base "true" margins and then add small noise.
    rng = np.random.default_rng(0)
    base = np.where(labels,  +2.5, -2.5).astype(float)   # perfect separability
    noise = rng.normal(0, 0.2, size=(2, labels.size))
    margins = base + noise

    storage = _DummyStorage(margins=margins, masks=masks, labels=labels)

    func(storage, split)

def test_ReturnCounterfactualOutputsBasic_Error_1():

    func = ReturnCounterfactualOutputsBasic()

    labels = np.array([0,1,1,1, 0,1,0,1], dtype=bool)

    # split indexes a subset of class 1 only
    split = np.array([1,0,1,1, 0,0,0,0], dtype=bool)

    # Masks: Split A must have at least some True (used in training) for ALL models;
    # Split B must have NO True anywhere.
    # We'll set: model0 saw two split_A samples, model1 saw one split_A; none saw split_B.
    masks = np.zeros((2, labels.size), dtype=bool)
    masks[0, split] = np.array([True, True, False])  # first two True among split_A positions
    masks[1, np.where(split)[0][0]] = True

    # create base "true" margins and then add small noise.
    rng = np.random.default_rng(0)
    base = np.where(labels,  +2.5, -2.5).astype(float)   # perfect separability
    noise = rng.normal(0, 0.2, size=(2, labels.size))
    margins = base + noise

    storage = _DummyStorage(margins=margins, masks=masks, labels=labels)

    with pytest.raises(ValueError, match='Split vector should index a subset of class 0 OR 1, right now it indexes both classes.'):
        func(storage, split)
    
def test_ReturnCounterfactualOutputsBasic_Error_3():

    func = ReturnCounterfactualOutputsBasic()

    labels = np.array([0,1,1,1, 0,1,0,1], dtype=bool)

    # split indexes a subset of class 1 only
    split = np.array([0,1,1,1, 0,0,0,0], dtype=bool)

    # Masks: Split A must have at least some True (used in training) for ALL models;
    # Split B must have NO True anywhere.
    # We'll set: model0 saw two split_A samples, model1 saw one split_A; none saw split_B.
    masks = np.zeros((2, labels.size), dtype=bool)
    masks[0, 0] = True # first two True among split_A positions
    masks[1, -1] = True

    # create base "true" margins and then add small noise.
    rng = np.random.default_rng(0)
    base = np.where(labels,  +2.5, -2.5).astype(float)   # perfect separability
    noise = rng.normal(0, 0.2, size=(2, labels.size))
    margins = base + noise

    storage = _DummyStorage(margins=margins, masks=masks, labels=labels)

    with pytest.raises(ValueError, match='Split_A was not used to train any models.'):
        func(storage, split)



def test_ReturnCounterfactualOutputsBasic_Error_4():

    func = ReturnCounterfactualOutputsBasic()

    labels = np.array([0,1,1,1, 0,1,0,1], dtype=bool)

    # split indexes a subset of class 1 only
    split = np.array([0,1,1,1, 0,0,0,0], dtype=bool)

    # Masks: Split A must have at least some True (used in training) for ALL models;
    # Split B must have NO True anywhere.
    # We'll set: model0 saw two split_A samples, model1 saw one split_A; none saw split_B.
    masks = np.zeros((2, labels.size), dtype=bool)
    masks[0, split] = np.array([True, True, False])  # first two True among split_A positions
    masks[1, np.where(split)[0][0]] = True
    masks[0,5] = True


    # create base "true" margins and then add small noise.
    rng = np.random.default_rng(0)
    base = np.where(labels,  +2.5, -2.5).astype(float)   # perfect separability
    noise = rng.normal(0, 0.2, size=(2, labels.size))
    margins = base + noise

    storage = _DummyStorage(margins=margins, masks=masks, labels=labels)

    with pytest.raises(ValueError, match='Split_B was used to train on. Only Split_A should be used for training.'):
        func(storage, split)
    #print(func._return_split_indices(split, storage, 1))