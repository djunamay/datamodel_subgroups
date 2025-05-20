# std-lib
from pathlib import Path
import warnings

# third-party
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import shuffle
from typing import Optional
from tqdm import tqdm
# project
from ..datasets import DatasetInterface
from ..datasamplers import MaskFactory
from ..datasamplers.random_generators import RandomGeneratorTCInterface
from ..models import ModelFactory
from ..models.base import SklearnClassifier
from ..utils.scoring import compute_margins
from ..datastorage.mask_margin import MaskMarginStorage
from ..datastorage.base import MaskMarginStorageInterface
import chz


@chz.chz
class TrainClassifiersArgs:
    """
    Lightweight container that bundles every input required by
    :func:`run_training_batch`.

    Parameters
    ----------
    dataset : DatasetInterface
        Provides the feature matrix and labels.
    mask_factory : MaskFactory
        Creates a boolean training-mask for each split.
    model_factory : ModelFactory
        Builds fresh classifier instances.
    n_models : int
        Number of classifiers (i.e., training splits) to run.
    random_generator : RandomGeneratorTCInterface
        Supplies all reproducibility seeds (mask, model, shuffle, …).  
        See the *RandomGenerator* docstring for the exact per-seed policy.
    in_memory : bool, default ``True``
        When *True*, results are kept in RAM and the function returns a
        :class:`MaskMarginStorage`.  When *False*, results are flushed to disk
        and the function returns the output path instead.
    path : Path | None, default ``None``
        Directory where on-disk results will be written if *in_memory* is
        *False*.  Ignored otherwise.
    """
    dataset: DatasetInterface = chz.field(default=None, doc='The dataset used for the experiment.')
    mask_factory: MaskFactory = chz.field(default=None, doc='Factory for generating masks. This will be used for training the classifier.')
    model_factory: ModelFactory = chz.field(default=None, doc='Factory for creating models. This will be used for training the classifier.')
    n_models: int = chz.field(default=None, doc='Number of models to build from ModelFactory. Each model will be trained on a different mask from MaskFactory.')
    in_memory: bool = chz.field(default=True, doc='Flag indicating whether to store results in memory.')
    path: Optional[str] = chz.field(default=None, doc='Path for storing results if not in memory.')
    random_generator: RandomGeneratorTCInterface = chz.field(default=None, doc='Random generator for training the classifier.')

def _make_storage(args: TrainClassifiersArgs, ds: DatasetInterface) -> MaskMarginStorageInterface:
    return MaskMarginStorage(
        args.n_models,
        ds.num_samples,
        ds.coarse_labels,
        args.mask_factory,
        args.in_memory,
        args.path,
        args.random_generator.batch_starter_seed,
    )

def _call_storage_warning(mask_shape, n_models):
    if mask_shape != n_models:
        warnings.warn(
            f"\nAn output array for this batch already exists and is of shape "
            f"{mask_shape}, which does not match the number of training "
            f"splits specified for this run ({n_models}). The existing array "
            f"will be completed.",
            UserWarning,
        )

def fit_single_classifier(features: NDArray[np.float32], labels: NDArray[bool], mask: NDArray[bool], model: SklearnClassifier, shuffle_seed: int):
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

def run_training_batch(args: TrainClassifiersArgs):
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
    storage = _make_storage(args, ds)
    _call_storage_warning(storage.masks.shape[0], args.n_models)

    for i, mask in tqdm(enumerate(storage.masks), desc='Training classifiers', total=args.n_models):
        if storage.is_filled(i):
            continue

        clf   = args.model_factory.build_model(seed=args.random_generator.model_build_seed)
        rng_s = args.random_generator.train_data_shuffle_seed
        margins, acc = fit_single_classifier(ds.features, ds.coarse_labels, mask, clf, rng_s)
        storage.fill_results(i, margins, acc)

    return storage if storage.in_memory else storage.path

if __name__ == "__main__":
    chz.nested_entrypoint(run_training_batch)
# python -m subgroups.experiments.train_classifiers dataset=subgroups.datasets.registry:gtex mask_factory=subgroups.datasamplers.mask_generators:fixed_alpha_mask_factory model_factory=subgroups.models.XgbFactory n_models=10    

