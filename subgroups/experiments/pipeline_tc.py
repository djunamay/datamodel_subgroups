import numpy as np
from tqdm import tqdm
from ..datastorage.experiment import Experiment
from .train_classifiers import TrainClassifiersArgs, run_training_batch
import os
import chz
from ..utils.configs import check_and_write_config

def pipeline_tc(experiment: Experiment, batch_size: int, batch_starter_seed: int=0, overwrite_config: bool=False, in_memory: bool=True) -> None:
    """
    Run a batch of training experiments for a given batch size and starter seed.
    The starter seed ensures that each model, each mask, and each training data shuffle are built from independent random seeds within and across each run.
    See `RandomGeneratorTC` for more details.
    """
    if not in_memory:
        path_to_config = os.path.join(experiment.path_to_results, "experiment_config.json")
        check_and_write_config(experiment, path_to_config, overwrite_config)

    random_generator = experiment.tc_random_generator(batch_starter_seed=batch_starter_seed)

    args = TrainClassifiersArgs(dataset=experiment.dataset, 
                                mask_factory=experiment.mask_factory, 
                                model_factory=experiment.model_factory, 
                                n_models=batch_size,
                                in_memory=in_memory,
                                path=experiment.path_to_classifier_outputs if not in_memory else None,
                                random_generator=random_generator,
                                npcs=experiment.npcs,
                                feature_selector=experiment.feature_selector)
    
    train_out = run_training_batch(args, batch_starter_seed)

    if in_memory:
        return train_out
    else:
        print(f"Train classifier output saved to {train_out}")


if __name__ == "__main__":
    chz.entrypoint(pipeline_tc)