from subgroups.datastorage.experiment import Experiment
from subgroups.counterfactuals.base import ReturnCounterfactualOutputInterface
from subgroups.datasamplers.mask_generators import mask_factory_counterfactuals
from subgroups.experiments.counterfactuals import CounterfactualArgs, run_counterfactual_for_one_split
from numpy.typing import NDArray

def pipeline_cfl(experiment: Experiment, split: NDArray[bool], n_models: int, batch_starter_seed: int, in_memory: bool, return_counterfactual_outputs: ReturnCounterfactualOutputInterface):

    counterfactual_mask_factory =mask_factory_counterfactuals(split=split, alpha=experiment.mask_factory.alpha)
    args = CounterfactualArgs(experiment=experiment,
                    nmodels=n_models,
                    batch_starter_seed=batch_starter_seed, 
                    CounterFactualMaskFactory=counterfactual_mask_factory,
                    in_memory=in_memory)
    training_output = run_counterfactual_for_one_split(args)
    
    if in_memory: # TODO: in_memory is already in the maskmarginstorage so dont need in_memory here
        return return_counterfactual_outputs(training_output, split=split)
    else:
        print(f"Train classifier output saved to {training_output}")


