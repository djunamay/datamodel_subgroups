from ..storage.experiment import SNRExperiment
from .compute_signal_to_noise import ComputeSNRArgsMultipleArchitectures, compute_snr_for_multiple_architectures
import os
import chz
from ..utils.configs import check_and_write_config

def pipeline_snr(experiment: SNRExperiment, batch_size: int, batch_starter_seed: int, overwrite_config: bool=False, in_memory: bool=True):
    """
    Compute the signal-to-noise ratio (SNR) for a number (batch_size) of model architectures.
    """
    if not in_memory:
        path_to_config = os.path.join(experiment.path_to_results, "snr_config.json")
        check_and_write_config(experiment, path_to_config, overwrite_config)
    
    random_generator = experiment.snr_random_generator(batch_starter_seed=batch_starter_seed)

    args = ComputeSNRArgsMultipleArchitectures(dataset=experiment.dataset, # TODO: this is too complicated (bad sign if have such complicated class names); if copy pasting, lots of words, etc something is wrong
                             in_memory=in_memory,
                             n_models=experiment.snr_n_models, 
                             n_passes=experiment.snr_n_passes, 
                             random_generator=random_generator,
                             path_to_results=experiment.path_to_snr_outputs if not in_memory else None,
                             n_architectures=batch_size,
                             model_factory_initializer=experiment.model_factory_initializer,
                             mask_factory_initializer=experiment.mask_factory_initializer,
                             stopping_condition=experiment.stopping_condition,
                             npcs_min=experiment.npcs_min,
                             npcs_max=experiment.npcs_max,
                             feature_selector=experiment.feature_selector)
    snr_out = compute_snr_for_multiple_architectures(args) # TODO this function can just take experiment as argument; this function just calls a function so don't need the whole pipeline_snr function

    if in_memory:
        return snr_out
    else:
        print(f"SNR output saved to {snr_out}")


if __name__ == "__main__":
    chz.entrypoint(pipeline_snr)