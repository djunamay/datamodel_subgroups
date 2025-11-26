# ── std-lib ───────────────────────────────────────────────────────────
from pathlib import Path
import os
from typing import Final, Type

# ── third-party ───────────────────────────────────────────────────────
import numpy as np
from tqdm import tqdm

# ── project ───────────────────────────────────────────────────────────
import chz

from ..datasets          import DatasetInterface
from ..datasamplers      import mask_factory_fn, mask_factory_init_fn
from ..datasamplers.random_generators import RandomGeneratorSNR
from ..classifiers import model_factory_fn, model_factory_init_fn
from ..utils.scoring     import compute_signal_noise
from ..utils.configs     import write_chz_class_to_json, append_float_ndjson
from .train_classifiers  import TrainClassifiersArgs, run_training_batch
from .stopping_condition import StoppingConditionInterface
from ..experiments.train_classifiers import fit_single_classifier, _make_storage
from ..datasamplers.feature_selectors import select_features_fn

@chz.chz
class ComputeSNRArgs:
    """
    Lightweight container that bundles every input required by
    :func:`compute_snr_for_one_architecture`.

    Parameters
    ----------
    dataset : DatasetInterface
        Provides the feature matrix and labels.
    mask_factory : subgroups.datasamplers.mask_factory_fn
        Creates a boolean training-mask for each split, using the method specified in the `MaskFactory`.
    model_factory : model_factory_fn
        Builds fresh classifiers instances, using the model architecture specified in the `ModelFactory`.
    random_generator : RandomGeneratorSNR
        Supplies all reproducibility seeds (mask, model, shuffle, …).  
        See the *RandomGenerator* docstring for the exact per-seed policy.
    n_models : int
        Number of classifiers (i.e., training splits) to run. 
    n_passes : int
        Number of times to build and train each of the `n_models` (i.e., maintaining the same training splits, but re-initializing the model and re-shuffling the training data).
    in_memory : bool, default ``True``
        When *True*, results are kept in RAM and the function returns a
        :class:`MaskMarginStorage`.  When *False*, results are flushed to disk
        and the function returns the output path instead.
    """
    dataset: DatasetInterface
    mask_factory: mask_factory_fn
    model_factory: model_factory_fn
    random_generator: RandomGeneratorSNR
    stopping_condition: StoppingConditionInterface
    npcs_min: int = 5
    npcs_max: int = 50

    n_models: int  = 20           # train-split count
    n_passes: int  = 15           # model re-initialisations per split
    in_memory: bool = True
    feature_selector: select_features_fn

@chz.chz
class ComputeSNRArgsMultipleArchitectures:
    """
    Lightweight container that bundles every input required by
    :func:`compute_snr_for_multiple_architectures`.

    Parameters
    ----------
    dataset : DatasetInterface
        Provides the feature matrix and labels.
    mask_factory_initializer : mask_factory_init_fn
        Initializes a new `MaskFactory` object, sampling from methods specified in the `MaskFactoryInitializer`.
    model_factory_initializer : model_factory_init_fn
        Initializes a new `ModelFactory` object, sampling from parameters specified in the `ModelFactoryInitializer`.
    random_generator : RandomGeneratorSNR
        Supplies all reproducibility seeds (mask, model, shuffle, …).  
        See the *RandomGenerator* docstring for the exact per-seed policy.
    n_models : int
        Number of classifiers (i.e., training splits) to run. 
    n_passes : int
        Number of times to build and train each of the `n_models` (i.e., maintaining the same training splits, but re-initializing the model and re-shuffling the training data).
    in_memory : bool, default ``True``
        When *True*, results are kept in RAM and the function returns a
        :class:`MaskMarginStorage`.  When *False*, results are flushed to disk
        and the function returns the output path instead.
    """
    dataset: DatasetInterface
    model_factory_initializer: model_factory_init_fn
    mask_factory_initializer: mask_factory_init_fn
    random_generator: Type[RandomGeneratorSNR]
    stopping_condition: StoppingConditionInterface

    n_models: int
    n_passes: int
    in_memory: bool
    npcs_min: int = 5
    npcs_max: int = 50

    n_architectures: int
    path_to_results: str

    feature_selector: select_features_fn


def _mk_train_args(cfg: ComputeSNRArgs) -> TrainClassifiersArgs:
    """Return the per-pass TrainClassifiersArgs object."""
    return TrainClassifiersArgs(
        dataset         = cfg.dataset,
        mask_factory    = cfg.mask_factory,
        model_factory   = cfg.model_factory,
        n_models        = cfg.n_models,
        random_generator= cfg.random_generator,
        in_memory       = True,
        feature_selector= cfg.feature_selector
    )

def _mk_snr_out(args: ComputeSNRArgsMultipleArchitectures) -> np.ndarray:
    if args.in_memory:
        snr_out = np.empty((args.n_architectures, args.dataset.num_samples))
    else:
        out_path = os.path.join(args.path_to_results, f"snr_batch_{args.random_generator.batch_starter_seed}.npy")
        snr_out = np.lib.format.open_memmap(out_path, dtype=np.float32, mode="w+", shape=(args.n_architectures, args.dataset.num_samples))
    return snr_out

def _mk_snr_args(args: ComputeSNRArgsMultipleArchitectures, mask_factory: mask_factory_fn, model_factory: model_factory_fn) -> ComputeSNRArgs:
    return ComputeSNRArgs(dataset=args.dataset, 
                         mask_factory=mask_factory,
                         model_factory=model_factory,
                         in_memory=args.in_memory, 
                         n_models=args.n_models, 
                         n_passes=args.n_passes, 
                         random_generator=args.random_generator,
                         stopping_condition=args.stopping_condition,
                         npcs_min=args.npcs_min,
                         npcs_max=args.npcs_max,
                         feature_selector=args.feature_selector)

def snr_inputs_for_one_architecture(args: ComputeSNRArgs) -> tuple[np.ndarray, float]:
    """
    Compute the signal-to-noise ratio (SNR) for a given ModelFactory and MaskFactory for each held-out sample.

    This function runs N passes of :func:`run_training_batch`, where each pass uses the same mask array but random model initializations and data shuffles.
    The resulting n_models x n_samples x n_passes array allows us to compute the signal-to-noise ratio for each held-out sample (i.e. how much of the variance in the margins is due to the mask vs model initialization and data shuffling).

    Parameters
    ----------
    args : ComputeSNRArgs
        Configuration and parameters for computing SNR, including dataset, mask factory,
        model factory, number of training splits, number of model initializations, and seeds.

    Returns
    -------
    np.ndarray
        Signal-to-noise ratio for each held-out sample (shape: [n_samples]).
    float
        Average test accuracy over all masks and all passes.
    """
    n_samples = args.dataset.num_samples
    masks    = np.empty((args.n_passes, args.n_models, n_samples), dtype=bool)
    margins  = np.empty((args.n_passes, args.n_models, n_samples), dtype=np.float32)

    if args.n_models < 50:
        bins = [(0, args.n_models)]
    else:
        bins = [(i, i + 50) for i in range(0, args.n_models, 50)]

    train_args = _mk_train_args(args)
    storage = _make_storage(train_args, args.dataset, args.random_generator.batch_starter_seed)
    n_pcs = args.random_generator._rngs_n_pcs_seed.integers(args.npcs_min, args.npcs_max)
    feature_indices = args.feature_selector(n_pcs)

    for bin in bins:

        start, stop = bin
                
        for p in tqdm(range(args.n_passes), desc=f"Passes"):

                for i in np.arange(start, stop):
                    mask = storage.masks[i]
                    clf   = args.model_factory(rng=args.random_generator.model_build_rng)
                    margins_temp, acc = fit_single_classifier(args.dataset.features[:, feature_indices], args.dataset.coarse_labels, mask, clf, args.random_generator.train_data_shuffle_rng)
                    storage.fill_results(i, margins_temp, acc)

                masks[p][start:stop]   = storage.masks[start:stop]
                margins[p][start:stop] = storage.margins[start:stop]

        print(f"Checking stopping condition after {stop} models")
        if args.stopping_condition.evaluate_stopping(margins[:,:stop,:], masks[:,:stop,:]):
            print(f"Stopping condition met after {stop} models")
            return margins[:,:stop,:], masks[:,:stop,:], storage.test_accuracies[:stop].mean(), n_pcs # TODO: test_accuracy is only averaged over 50 masks (1 bin - i.e. last storage). Might want to increase this to get a better estimate.
        
    return margins, masks, storage.test_accuracies.mean(), n_pcs



def compute_snr_for_multiple_architectures(args: ComputeSNRArgsMultipleArchitectures) -> np.ndarray:
    """
    This function runs :func:`compute_snr_for_one_architecture` for randomly sampled ModelFactory and MaskFactory (collectively referred to as "architectures") objects.
    It returns an array of shape [n_architectures, n_samples] containing the signal-to-noise ratio for each held-out sample and each architecture.

    Parameters
    ----------
    args : ComputeSNRArgsMultipleArchitectures
        Configuration and parameters for computing SNR, including dataset, mask factory initializer, model factory initializer, number of training splits, number of model initializations, and seeds.

    Returns
    -------
    np.ndarray
        Signal-to-noise ratio for each held-out sample and each architecture (shape: [n_architectures, n_samples]).
    """
    snr_out = _mk_snr_out(args)
    n_architectures = snr_out.shape[0]

    print(f"Batch {args.random_generator.batch_starter_seed}:")

    for i in range(n_architectures):
        print(f"Computing signal-to-noise ratio for architecture {i}", end="\n" + "-"*len(f"Computing signal-to-noise ratio for architecture {i}") + "\n")
        new_mask_factory = args.mask_factory_initializer(args.random_generator.mask_factory_rng)
        new_model_factory = args.model_factory_initializer(args.random_generator.model_factory_rng)
        margins, masks, test_accuracy, n_pcs = snr_inputs_for_one_architecture(_mk_snr_args(args, new_mask_factory, new_model_factory))
        snr = compute_signal_noise(margins, masks)
        snr_out[i] = snr

        if not args.in_memory:
            write_chz_class_to_json(new_model_factory, os.path.join(args.path_to_results, f"model_factory_{args.random_generator.batch_starter_seed}.json"), indent=None)
            write_chz_class_to_json(new_mask_factory, os.path.join(args.path_to_results, f"mask_factory_{args.random_generator.batch_starter_seed}.json"), indent=None)
            append_float_ndjson(n_pcs, os.path.join(args.path_to_results, f"n_pcs_{args.random_generator.batch_starter_seed}.json"))
            append_float_ndjson(test_accuracy, os.path.join(args.path_to_results, f"test_accuracy_{args.random_generator.batch_starter_seed}.json"))
            append_float_ndjson(np.nanmean(snr), os.path.join(args.path_to_results, f"snr_{args.random_generator.batch_starter_seed}.json")) 

    if args.in_memory:
        return snr_out
    else:
        return args.path_to_results

if __name__ == "__main__":
    chz.nested_entrypoint(snr_inputs_for_one_architecture)