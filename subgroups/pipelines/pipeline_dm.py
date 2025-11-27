import chz
import os
import numpy as np

from ..utils.configs import check_and_write_config
from ..storage.experiment import Experiment


def pipeline_dm(experiment: Experiment, batch_starter_seed: int, overwrite_config: bool=False, in_memory: bool=True, dm_n_train: int=1000, dm_n_test: int=1000) -> None:
    """
    Fit datamodels to a given set of indices (samples).
    """

    if not in_memory:
        path_to_config = os.path.join(experiment.path_to_results, "experiment_config.json")
        check_and_write_config(experiment, path_to_config, overwrite_config)
    
    if experiment.sample_selector is None:
        indices = np.arange(experiment.dataset.num_samples)
    else:
        indices = experiment.sample_selector(batch_starter_seed)

    
    dm_out = experiment.datamodels_pipeline.fit_datamodels(indices=indices, 
                                                           seed=batch_starter_seed, 
                                                           n_train=dm_n_train,
                                                           n_test=dm_n_test,
                                                           in_memory=in_memory)
    
    if in_memory:
        return dm_out
    else:
        print(f"DM output saved to {dm_out}")


if __name__ == "__main__":
    chz.entrypoint(pipeline_dm)