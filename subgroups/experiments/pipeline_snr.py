import numpy as np
from tqdm import tqdm
from ..datastorage.experiment import Experiment
from .compute_signal_to_noise import ComputeSNRArgsMultipleArchitectures, compute_snr_for_multiple_architectures
import os
import chz
from ..utils.configs import check_and_write_config

def pipeline_snr(experiment: Experiment, batch_size: int, batch_starter_seed: int, overwrite_config: bool=False):
    """
    Compute the signal-to-noise ratio (SNR) for a number (batch_size) of model architectures.
    """
    if not experiment.in_memory:
        path_to_config = os.path.join(experiment.path_to_results, "experiment_config.json")
        check_and_write_config(experiment, path_to_config, overwrite_config)
    
    random_generator = experiment.snr_random_generator(batch_starter_seed=batch_starter_seed)

    args = ComputeSNRArgsMultipleArchitectures(dataset=experiment.dataset, 
                             in_memory=experiment.in_memory, 
                             n_models=experiment.snr_n_models, 
                             n_passes=experiment.snr_n_passes, 
                             random_generator=random_generator,
                             path_to_results=experiment.path_to_snr_outputs if not experiment.in_memory else None,
                             n_architectures=batch_size,
                             model_factory_initializer=experiment.model_factory_initializer,
                             mask_factory_initializer=experiment.mask_factory_initializer,
                             stopping_condition=experiment.stopping_condition) 
    
    snr_out = compute_snr_for_multiple_architectures(args)

    if experiment.in_memory:
        return snr_out
    else:
        print(f"SNR output saved to {snr_out}")


if __name__ == "__main__":
    chz.entrypoint(pipeline_snr)