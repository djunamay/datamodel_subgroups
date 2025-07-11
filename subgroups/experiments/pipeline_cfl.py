from .run_counterfactuals import run_counterfactuals
import chz
import numpy as np
from subgroups.utils.random import fork_rng
from ..counterfactuals.base import CounterfactualInputsInterface, CounterfactualEvaluationInterface
from ..datastorage.experiment import Experiment
import os

@chz.chz
class CounterfactualExperimentArgs:

    counterfactual_inputs: CounterfactualInputsInterface = chz.field(default=None, doc='The dataset used for the experiment.')
    counterfactual_estimator: CounterfactualEvaluationInterface = chz.field(default=None, doc='Factory for generating masks. This will be used for training the classifier.')
    experiment_seed: int = chz.field(default=None, doc='Factory for creating models. This will be used for training the classifier.')
    n_iter: int = chz.field(default=None, doc='Number of models to build from ModelFactory. Each model will be trained on a different mask from MaskFactory.')
    n_clusters: int = chz.field(default=None, doc='Flag indicating whether to store results in memory.')
    in_memory: bool = chz.field(default=True, doc='Flag indicating whether to store results in memory.')
    path: str = chz.field(default=None, doc='Path for storing results if not in memory.')


def run_pipeline_counterfactuals(args: CounterfactualExperimentArgs):
    
    children = fork_rng(np.random.default_rng(args.experiment_seed), 3)
    results = run_counterfactuals(args.counterfactual_inputs.to_counterfactual_inputs, args.counterfactual_estimator, n_iter=args.n_iter, n_clusters=args.n_clusters, random_state=children[0], model_rng=children[1], shuffle_rng=children[2])

    if args.in_memory:
        return results
    else:
        results.to_csv(args.path, index=False)
        return args.path
    
def pipeline_counterfactuals(experiment: Experiment, experiment_seed: int, n_iter: int, n_clusters: int, in_memory: bool, group_1: bool=True, sampling_size_factor: int=2):

    if not in_memory:
        if group_1:
            group_str = "group_1"
        else:
            group_str = "group_2"
        out_path = os.path.join(experiment.path, experiment.experiment_name, "clustering_outputs", f"batch_{experiment_seed}_nclusters_{n_clusters}_{group_str}_counterfactual_results.csv")
    else:
        out_path = None
    
    counterfactual_inputs = experiment.counterfactual_inputs(path_to_features=experiment.dataset.path_to_data, 
                                                       path_to_weights=os.path.join(experiment.path, experiment.experiment_name, "datamodel_outputs"), 
                                                       dataset=experiment.dataset,
                                                       group_1=group_1)


    counterfactual_estimator = experiment.counterfactual_estimator(features=experiment.dataset.features[:,experiment.feature_selector.feature_indices(n_pcs=experiment.npcs)],
                                                        coarse_labels=experiment.dataset.coarse_labels,
                                                        train_size=int((experiment.mask_factory.alpha*experiment.dataset.num_samples)/sampling_size_factor), 
                                                        test_size=int((experiment.counterfactual_test_fraction*experiment.dataset.num_samples)/sampling_size_factor),
                                                        classifier=experiment.model_factory,
                                                        group_1=group_1)

    args = CounterfactualExperimentArgs(counterfactual_inputs=counterfactual_inputs, 
                                 counterfactual_estimator=counterfactual_estimator, 
                                 experiment_seed=experiment_seed, n_iter=n_iter, 
                                 n_clusters=n_clusters, 
                                 in_memory=in_memory, 
                                 path=out_path)
    
    results = run_pipeline_counterfactuals(args)

    if in_memory:
        return results
    else:
        print(f"Counterfactual results saved to {results}")
    
    


if __name__ == "__main__":
    chz.entrypoint(pipeline_counterfactuals)