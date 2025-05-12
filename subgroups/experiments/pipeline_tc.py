import numpy as np
from tqdm import tqdm
from ..datastorage.experiment import Experiment
from .train_classifiers import TrainClassifiersArgs, train_classifiers
import os
import chz
from ..utils.configs import check_and_write_config
from ..utils.randomness import generate_rngs_from_seed

def pipeline_tc(experiment: Experiment, batch_size: int, batch_starter_seed: int=0, overwrite: bool=False):
    """
    Compute the SNR for a given index and batch size.
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
    
    train_out = train_classifiers(args)

    if experiment.in_memory:
        return train_out
    else:
        print(f"Train classifier output saved to {train_out}")


if __name__ == "__main__":
    chz.entrypoint(pipeline_tc)