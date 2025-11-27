# Standard library
import os
from typing import Union

# Third-party libraries
import chz
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error
from sklearn.utils import shuffle
from tqdm import tqdm

# Local/project imports
from subgroups.storage.experiment import Experiment
from ..storage.training import DatamodelStorage, MaskMarginStorage
from ..utils.configs import check_and_write_config
from ..utils.random import fork_rng

# Type aliases
Array = Union[np.ndarray, np.memmap]

def get_samples_for_model(masks, margins, seed_shuffle, sample_index: int, n_train: int, n_test: int = None):
    """
    Get the samples for the model.
    For a given sample, exclude any models for fitting in which that sample was included in the training set.
    """
    mask = masks[:, sample_index]
    index = ~mask
    y = margins[index, sample_index]
    X = masks[index]

    if n_test is not None:
        X, y = shuffle(X[:n_train + n_test], y[:n_train + n_test],
                       random_state=seed_shuffle)  # reduce total matrix size for shuffling
    else:
        X, y = shuffle(X, y, random_state=seed_shuffle)

    return X, y


def fit_datamodels_batch(experiment : Experiment, mask_margin_storage : MaskMarginStorage, batch_size : int, n_train : int, n_test : int=None, batch_starter_seed : int=0, in_memory : bool=True, overwrite_config : bool=False):

    if not in_memory:
        check_and_write_config(experiment, os.path.join(experiment.path_to_results, "experiment_config.json"), overwrite_config)

    sample_indices = np.arange(batch_starter_seed * batch_size, batch_starter_seed * batch_size + batch_size)

    random_generator = experiment.tc_random_generator(batch_starter_seed=batch_starter_seed)
    storage = DatamodelStorage(n_models=batch_size,
                               n_samples=experiment.dataset.num_samples,
                               path_to_outputs=os.path.join(experiment.path_to_datamodel_outputs, f"batch_{batch_starter_seed}") if not in_memory else None)

    build_model_rngs_children = fork_rng(random_generator.model_build_rng, batch_size)
    train_data_shuffle_rngs_children = fork_rng(random_generator.train_data_shuffle_rng, batch_size)

    for i, sample_index in tqdm(enumerate(sample_indices), total=batch_size):

        if storage.is_filled(i):
            continue

        X, y = get_samples_for_model(mask_margin_storage.masks, mask_margin_storage.margins, train_data_shuffle_rngs_children[sample_index].integers(0, 2 ** 32 - 1), sample_index, n_train, n_test)
        model = experiment.datamodel_factory(seed=build_model_rngs_children[sample_index].integers(0, 2 ** 32 - 1))

        model.fit(X[:n_train], y[:n_train])

        stop = n_train + n_test if n_test is not None else X.shape[0]
        y_test, y_hat = y[n_train:stop], model.predict(X[n_train:stop])

        storage.fill_results(i, weights=model.coef_, biases=model.intercept_, p_correlations=pearsonr(y_test, y_hat)[0], s_correlations=spearmanr(y_test, y_hat)[0], rmses=root_mean_squared_error(y_test, y_hat), sample_indices=sample_index)

    return storage if in_memory else print(f"Fit datamodel output saved to {experiment.path_to_datamodel_outputs}")

if __name__ == "__main__":
    chz.entrypoint(fit_datamodels_batch)