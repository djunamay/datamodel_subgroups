
from ..experiments.train_classifiers import TrainClassifiersArgs, run_training_batch
from ..datasamplers.base import MaskFactory
from ..datastorage.experiment import Experiment

import chz 

@chz.chz
class CounterfactualArgs:
    experiment: Experiment = chz.field(default=None, doc='Factory for generating masks. This will be used for training the classifier.')
    nmodels: int = chz.field(default=None, doc='Number of models to build from ModelFactory per counterfactual experiment. Each model will be trained on a different mask from MaskFactory.')
    batch_starter_seed: int
    CounterFactualMaskFactory: MaskFactory
    in_memory: bool
    
def run_counterfactual_for_one_split(CounterfactualArgs):
    experiment = CounterfactualArgs.experiment
    random_generator = experiment.tc_random_generator(batch_starter_seed=CounterfactualArgs.batch_starter_seed)

    args = TrainClassifiersArgs(dataset=experiment.dataset, 
                                    mask_factory=CounterfactualArgs.CounterFactualMaskFactory,  
                                    model_factory=experiment.model_factory, 
                                    n_models=CounterfactualArgs.nmodels,
                                    in_memory=CounterfactualArgs.in_memory, 
                                    path=experiment.path_to_classifier_outputs if not experiment.in_memory else None, 
                                    random_generator=random_generator,
                                    npcs=experiment.npcs,
                                    feature_selector=experiment.feature_selector)
        
    train_out = run_training_batch(args, CounterfactualArgs.batch_starter_seed)

    if CounterfactualArgs.in_memory:
        return train_out
    else:
        print(f"Train classifier output saved to {train_out}")