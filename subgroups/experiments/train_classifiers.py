from ..datasets import DatasetInterface
from ..datasamplers import MaskFactory
from ..models import ModelFactory
from ..utils.scoring import compute_margins
from ..datastorage.base import MaskMarginStorage
from typing import Optional
from pathlib import Path
import chz
import numpy as np
from ..models.base import SklearnClassifier
from numpy.typing import NDArray
from sklearn.utils import shuffle

@chz.chz
class TrainClassifiersArgs:
    """
    Configuration arguments for training classifiers.

    Attributes
    ----------
    dataset : DatasetInterface
        Interface for accessing the dataset.
    mask_factory : MaskFactory
        Factory for generating training sample masks.
    model_factory : ModelFactory
        Factory for creating model instances.
    n_models : int
        Number of models to train.
    in_memory : bool, default=True
        Flag indicating whether to store results in memory.
    path : Optional[Path], optional
        Path for storing results if not in memory.
    mask_seed : Optional[int], optional
        Seed for mask generation reproducibility.
        This seed is used to initialize the random number generator for mask generation, resulting in a determistic sequence of masks.
    model_seed : Optional[int], optional
        Seed for model initialization reproducibility.
        This seed is used to initialize the random number generator for model initialization, resulting in a determistic sequence of models.
    """
    dataset: DatasetInterface
    mask_factory: MaskFactory
    model_factory: ModelFactory
    n_models: int
    in_memory: bool = True
    path: Optional[Path] = None
    mask_seed: Optional[int] = None
    model_seed: Optional[int] = None

def train_one_classifier(features: NDArray[np.float32], labels: NDArray[bool], mask: NDArray[bool], model: SklearnClassifier, shuffle_seed: int):
    """
    Train a single classifier and compute margins and test accuracy.

    Parameters
    ----------
    features : NDArray[np.float32]
        Feature matrix for all samples (shape: [n_samples, n_features]).
    labels : NDArray[bool]
        Binary labels for all samples (shape: [n_samples,]).
    mask : NDArray[bool]
        Boolean mask indicating training samples (shape: [n_samples,]).
    model : SklearnClassifier
        Classifier model to be trained.
    shuffle_seed : int
        Seed for shuffling the training samples.

    Returns
    -------
    margins : NDArray[float]
        Computed margins for each sample (shape: [n_samples,]).
    test_accuracy : float
        Accuracy of the model on the test set.
    """
    features_shuffled, labels_shuffled = shuffle(features[mask], labels[mask], random_state=shuffle_seed) 
    model.fit(features_shuffled, labels_shuffled) 
    test_accuracy = model.score(features[~mask], labels[~mask])
    margins = compute_margins(model.predict_proba(features)[:,1], labels)
    return margins, test_accuracy

def train_classifiers(args: TrainClassifiersArgs):
    """
    Train multiple classifiers and store the results in a mask margin storage object.

    Parameters
    ----------
    args : TrainClassifiersArgs
        Configuration and parameters for training classifiers, including dataset, 
        mask factory, model factory, number of models, and storage options.

    Returns
    -------
    MaskMarginStorage or Path
        If `in_memory` is True, returns the MaskMarginStorage object containing 
        the training results. Otherwise, returns the path to the stored results.
    """
    ds = args.dataset
    model_factory = args.model_factory
    n_train_splits = args.n_models
    mask_margin_storage = MaskMarginStorage(n_train_splits, ds.num_samples, ds.coarse_labels, args.mask_factory, args.in_memory, args.path, args.mask_seed) # TODO: move this outside of the train_classifier function but then would need some way to "flush it"
    rng = np.random.default_rng(args.model_seed)

    for i in range(n_train_splits):

        if mask_margin_storage.is_filled(i):
            continue
        else: 
            model = model_factory.build_model(seed=rng.integers(0, 2**32 - 1))
            margins, test_accuracy = train_one_classifier(ds.features, ds.coarse_labels, mask_margin_storage.masks[i], model, rng.integers(0, 2**32 - 1))
            mask_margin_storage.fill_results(i, margins, test_accuracy)

    if mask_margin_storage.in_memory:
        return mask_margin_storage
    else:
        return mask_margin_storage.path

if __name__ == "__main__":
    chz.nested_entrypoint(train_classifiers)



# python -m subgroups.experiments.train_classifiers dataset=subgroups.datasets.registry:gtex mask_factory=subgroups.datasamplers.mask_generators:fixed_alpha_mask_factory model_factory=subgroups.models.XgbFactory n_models=10    