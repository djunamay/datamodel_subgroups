import chz
import os
import numpy as np

from ..utils.configs import check_and_write_config
from ..datastorage.experiment import Experiment


def pipeline_dm(experiment: Experiment, batch_starter_seed: int, overwrite_config: bool=False):

    if not experiment.in_memory:
        path_to_config = os.path.join(experiment.path_to_results, "experiment_config.json")
        check_and_write_config(experiment, path_to_config, overwrite_config)
    
    if experiment.indices_to_fit is None:
        indices = np.arange(experiment.n_samples)
    else:
        indices = experiment.indices_to_fit(batch_starter_seed)

    
    dm_out = experiment.datamodels_pipeline.fit_datamodels(indices, rng = np.random.default_rng(batch_starter_seed), 
                                                           path_to_outputs=experiment.path_to_datamodel_outputs + f"/batch_{str(batch_starter_seed)}" if not experiment.in_memory else None, 
                                                           in_memory=experiment.in_memory,
                                                           n_train=experiment.dm_n_train,
                                                           n_test=experiment.dm_n_test)
    
    if experiment.in_memory:
        return dm_out
    else:
        print(f"DM output saved to {dm_out}")


if __name__ == "__main__":
    chz.entrypoint(pipeline_dm)