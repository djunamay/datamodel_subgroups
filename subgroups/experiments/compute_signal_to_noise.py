from ..models import ModelFactory   
from ..datasets import DatasetInterface
from ..utils.scoring import compute_signal_noise
from ..models import ModelFactory
from ..datasamplers import MaskFactory
import chz
import numpy as np
from .train_classifiers import TrainClassifiersArgs, train_classifiers

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
    mask_factory: MaskFactory
    model_factory: ModelFactory  
    mask_seed: int

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

    for i in range(args.n_model_inits):
        classifier_args = TrainClassifiersArgs(dataset=args.dataset, 
                     mask_factory=args.mask_factory, 
                     model_factory=args.model_factory, 
                     n_models=args.n_train_splits,
                     in_memory=True, 
                     mask_seed=args.mask_seed,
                     model_seed=i
                     )

        out = train_classifiers(classifier_args)
        out_masks[:,:,i] = out.masks
        out_margins[:,:,i] = out.margins
    
    return compute_signal_noise(out_margins, out_masks)