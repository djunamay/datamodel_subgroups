from ..models import ModelFactory   
from ..datasets import DatasetInterface
from ..utils.scoring import compute_signal_noise
from ..models import ModelFactory, ModelFactoryInitializer
from ..datasamplers import MaskFactory
import chz
import numpy as np
from .train_classifiers import TrainClassifiersArgs, train_classifiers
import os
from tqdm import tqdm
from numpy.random import Generator
from ..utils.randomness import generate_rngs_from_seed
@chz.chz
class ComputeSNRArgs:
    """
    Configuration arguments for computing the signal-to-noise ratio (SNR).

    Attributes
    ----------
    dataset : DatasetInterface
        Interface for accessing the dataset.
    mask_factory : MaskFactory
        Factory for generating training sample masks.
    model_factory : ModelFactory
        Factory for creating model instances.
    in_memory : bool
        Flag indicating whether to store results in memory.
    n_train_splits : int
        Number of training splits.
    n_model_inits : int
        Number of model initializations.
    mask_seed : int
        Seed for mask generation reproducibility.
    """
    dataset: DatasetInterface
    mask_factory: MaskFactory
    model_factory: ModelFactory
    in_memory: bool
    n_train_splits: int
    n_model_inits: int
    rngs: dict[str, Generator]

@chz.chz
class ComputeSNRArgsMultipleArchitectures:
    """
    Configuration arguments for computing the signal-to-noise ratio (SNR) for multiple architectures.

    Attributes
    ----------
    dataset : DatasetInterface
        Interface for accessing the dataset.
    mask_factory : MaskFactory
        Factory for generating training sample masks.
    model_factory : ModelFactory
        Factory for creating model instances.
    in_memory : bool
        Flag indicating whether to store results in memory.
    n_train_splits : int
        Number of training splits.
    n_model_inits : int
        Number of model initializations.
    mask_seed : int
        Seed for mask generation reproducibility.
    """
    dataset: DatasetInterface
    mask_factory: MaskFactory
    model_factory: ModelFactory
    in_memory: bool
    n_train_splits: int
    n_model_inits: int
    path_to_results: str
    n_architectures: int
    model_factory_initializer: ModelFactoryInitializer
    batch_starter_seed: int

    
def compute_snr_for_one_architecture(args: ComputeSNRArgs) -> np.ndarray:
    """
    Compute the signal-to-noise ratio (SNR) for a given ModelFactory and MaskFactory for each held-out sample.

    This function initializes and trains 'n_model_inits' classifiers for each 'n_train_splits' mask and collects the resulting masks and margins, and computes the SNR.

    Parameters
    ----------
    args : ComputeSNRArgs
        Configuration and parameters for computing SNR, including dataset, mask factory,
        model factory, number of training splits, number of model initializations, and seeds.

    Returns
    -------
    np.ndarray
        Signal-to-noise ratio for each held-out sample (shape: [n_samples]).
    """
    out_masks = np.empty((args.n_train_splits, args.dataset.num_samples, args.n_model_inits), dtype=bool)
    out_margins = np.empty((args.n_train_splits, args.dataset.num_samples, args.n_model_inits), dtype=np.float32)

    mask_starter_seed = args.rngs['mask_starter_rng'].integers(0, 2**32 - 1)
    model_starter_rng = args.rngs['model_starter_rng']

    for i in range(args.n_model_inits):
        classifier_args = TrainClassifiersArgs(dataset=args.dataset, 
                     mask_factory=args.mask_factory, 
                     model_factory=args.model_factory, 
                     n_models=args.n_train_splits,
                     in_memory=True,
                     rngs={**generate_rngs_from_seed(mask_starter_seed, ['mask_rng']), **generate_rngs_from_seed(model_starter_rng.integers(0, 2**32 - 1), ['model_rng','shuffle_rng'])})

        out = train_classifiers(classifier_args)
        out_masks[:,:,i] = out.masks
        out_margins[:,:,i] = out.margins
    
    return compute_signal_noise(out_margins, out_masks)

def compute_snr_for_multiple_architectures(args: ComputeSNRArgsMultipleArchitectures) -> np.ndarray:

    if args.in_memory:
        snr_out = np.empty((args.n_architectures, args.dataset.num_samples))
    else:
        out_path = os.path.join(args.path_to_results, f"snr_batch_{args.n_architectures}.npy")
        snr_out = np.lib.format.open_memmap(out_path, dtype=np.float32, mode="w+", shape=(args.n_architectures, args.dataset.num_samples))

    rngs = generate_rngs_from_seed(args.batch_starter_seed, ['mask_starter_rng', 'model_starter_rng', 'architecture_rng'])

    for i in tqdm(range(snr_out.shape[0])):
        model_factory = args.model_factory_initializer.build_model_factory(rngs['architecture_rng'].integers(0, 2**32 - 1))
        SNRargs = ComputeSNRArgs(dataset=args.dataset, 
                             mask_factory=args.mask_factory, # TODO: change to new mask factory each iteration - use index
                             model_factory=model_factory, # TODO: change to new model factory each iteration - use index
                             in_memory=args.in_memory, 
                             n_train_splits=args.n_train_splits, 
                             n_model_inits=args.n_model_inits, 
                             rngs=rngs)
        snr = compute_snr_for_one_architecture(SNRargs)
        snr_out[i] = snr

    if args.in_memory:
        return snr_out
    else:
        return args.path_to_results

if __name__ == "__main__":
    chz.nested_entrypoint(compute_snr_for_one_architecture)