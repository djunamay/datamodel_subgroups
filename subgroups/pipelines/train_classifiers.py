# Standard library
import os
import warnings

# Third-party libraries
import chz
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

# Local/project imports
from ..models import SklearnClassifier
from ..storage.experiment import Experiment
from ..storage.training import MaskMarginStorage
from ..utils.configs import check_and_write_config
from ..utils.random import fork_rng
from ..utils.scoring import compute_margins


def fit_single_classifier(features: NDArray[np.float32], labels: NDArray[bool], mask: NDArray[bool], model: SklearnClassifier, shuffle_rng: np.random.Generator):
    """
    Train a single models and compute margins and test accuracy.

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
    random_state = np.random.RandomState(shuffle_rng.bit_generator)

    features_shuffled, labels_shuffled = shuffle(features[mask], labels[mask], random_state=random_state) 
    model.fit(features_shuffled, labels_shuffled) 

    tl, tf = labels[~mask], features[~mask]
    _, test_features, _, test_labels  = train_test_split(tf, tl, stratify=tl, test_size=min(tl.sum(), (~tl).sum()), random_state=random_state) # balance the test data by label

    test_accuracy = model.score(test_features, test_labels)
    margins = compute_margins(model.predict_proba(features)[:,1], labels)
    return margins, test_accuracy

def run_training_batch(experiment: Experiment, batch_size: int, batch_starter_seed: int=0, in_memory: bool=True, overwrite_config: bool=False):
    """
    Train multiple models and store the results in a mask margin storage object.

    Parameters
    ----------
    args : TrainClassifiersArgs
        Configuration and parameters for training models, including dataset,
        mask factory, model factory, number of models, and storage options.

    Returns
    -------
    MaskMarginStorage or Path
        If `in_memory` is True, returns the MaskMarginStorage object containing 
        the training results. Otherwise, returns the path to the stored results.
    """
    if not in_memory:
        check_and_write_config(experiment, os.path.join(experiment.path_to_results, "experiment_config.json"), overwrite_config)

    random_generator = experiment.tc_random_generator(batch_starter_seed=batch_starter_seed)
    storage = MaskMarginStorage(n_models=batch_size,
                                n_samples=experiment.dataset.num_samples,
                                labels=experiment.dataset.coarse_labels,
                                mask_factory=experiment.mask_factory,
                                path_to_outputs=os.path.join(experiment.path_to_classifier_outputs, f"batch_{batch_starter_seed}") if not in_memory else None,
                                rng= random_generator.mask_rng)

    if storage.masks.shape[0] != batch_size:
        warnings.warn(
            f"\nAn output array for this batch already exists and is of shape "
            f"{storage.masks.shape[0]}, which does not match the number of training "
            f"splits specified for this run ({batch_size}). The existing array "
            f"will be completed.",
            UserWarning,
        )

    build_model_rngs_children = fork_rng(random_generator.model_build_rng, batch_size)
    train_data_shuffle_rngs_children = fork_rng(random_generator.train_data_shuffle_rng, batch_size)

    features = experiment.dataset.features[:, experiment.feature_selector(experiment.npcs)]
    for i, mask in tqdm(enumerate(storage.masks), desc='Training models', total=batch_size):
        
        if storage.is_filled(i):
            continue

        clf = experiment.model_factory(rng=build_model_rngs_children[i])
        margins, acc = fit_single_classifier(features, experiment.dataset.coarse_labels, mask, clf, train_data_shuffle_rngs_children[i])
        storage.fill_results(i, margins=margins, test_accuracies=acc)

    return storage if in_memory else print(f"Train classifier output saved to {experiment.path_to_classifier_outputs}")

if __name__ == "__main__":
    chz.entrypoint(run_training_batch)

