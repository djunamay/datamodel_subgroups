import numpy as np
from tqdm import tqdm
from ..datastorage.experiment import Experiment
from .train_classifiers import TrainClassifiersArgs, run_training_batch
import os
import chz
from ..utils.configs import check_and_write_config

def pipeline_tc(experiment: Experiment, batch_size: int, batch_starter_seed: int=0, overwrite: bool=False):
    """
    Run a batch of training experiments for a given batch size and starter seed.
    The starter seed ensures that each model, each mask, and each training data shuffle are built from independent random seeds within and across each run.
    See `RandomGeneratorTCInterface` for more details.
    """
    if not experiment.in_memory:
        path_to_config = os.path.join(experiment.path_to_results, "experiment_config.json")
        check_and_write_config(experiment, path_to_config, overwrite)

    random_generator = experiment.tc_random_generator(batch_starter_seed=batch_starter_seed)

    args = TrainClassifiersArgs(dataset=experiment.dataset, 
                                mask_factory=experiment.mask_factory, 
                                model_factory=experiment.model_factory, 
                                n_models=batch_size,
                                in_memory=experiment.in_memory, 
                                path=experiment.path_to_classifier_outputs if not experiment.in_memory else None, 
                                random_generator=random_generator)
    
    train_out = run_training_batch(args)

    if experiment.in_memory:
        return train_out
    else:
        print(f"Train classifier output saved to {train_out}")


if __name__ == "__main__":
    chz.entrypoint(pipeline_tc)