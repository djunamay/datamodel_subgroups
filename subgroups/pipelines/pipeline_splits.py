from ..storage.experiment import Experiment
from ..storage.splits import SplitStorage, Split
from ..splits.base import SplitFactoryInterface
from ..counterfactuals.base import ReturnCounterfactualOutputInterface
from ..splits.base import ProcessExperimentForSplitsInterface, ReturnBestSplitInterface

from numpy.typing import NDArray 
import numpy as np

import chz

# the chz classes define a job -> this is what I want to do
# then it's taken by something (can read it at any time, can then create classes that then can mutate to do the job)
# and classes that just take in to out are not classes, they are functions
# classes are things that store the data alongside ways to operate on the data
# any classes the user interacts with should be a chz class
# a way for the user to configure what they want (define the job they are trying to do)

def pipeline_split(experiment: Experiment, 
                   process_input_data: ProcessExperimentForSplitsInterface, #TODO: replace this with Callable[[Experiment], NDArray] 
                   Splitter: SplitFactoryInterface, 
                   return_counterfactual_outputs: ReturnCounterfactualOutputInterface, 
                   ReturnBestSplit: ReturnBestSplitInterface, 
                   K: NDArray[float], 
                   n_models: int, 
                   in_memory: bool,
                   split_class_1: bool,
                   score_thresh: float):

    """
    Returns instance of class SplitStorage containing splits for one group of experiment.dataset.coarse_labels based on methods defined in process_input_data, Splitter, return_counterfactual_outputs, and ReturnBestSplit.

    Parameters
    ----------
    experiment : Experiment
    process_input_data : ProcessExperimentForSplitsInterface
            Transformation to apply to experiment to use as input for Splitter.
    Splitter : SplitFactoryInterface
            Factory that generates splits from processed input data based on some strategy.
    return_counterfactual_outputs : ReturnCounterfactualOutputInterface
            Function that summarizes counterfactual training outputs. Must have 'score' attribute.
    ReturnBestSplit : ReturnBestSplitInterface
    K : NDArray[float]
            Array of parameters of values used by SplitFactoryInterface to generate splits.
    n_models : int
            Number of models to perform counterfactual pipelines on.
    in_memory : bool
            If True, run computation without writing intermediate results to disk.

    Returns
    -------
    instance of class SplitStorage

    """
    A = process_input_data(experiment) # NO TEST

    if split_class_1:
        coarse_labels = experiment.dataset.coarse_labels
    else: 
        coarse_labels = np.invert(experiment.dataset.coarse_labels)

    IDs = np.argwhere(coarse_labels).reshape(-1) # TESTED
    split_storage = SplitStorage(splits={'split_0':Split(IDs)}) # TESTED

        # start with assignment; all datapoints are label 0
        # two sets; working labels and done labels
        # working_on starts with 0
        # done is empty
        # then do the while for loop, while something working on, pick it, run the split
        # if find that there is no good split, keep labeling as is and put to done instead
        # if find a good split 
        # then return an array  
        # implement as standard scikit learn interface / look at examples that may return additional information       
    while len(split_storage.unfinished_splits)!=0:
        current_keys = list(split_storage.splits.keys())

        for split_name in current_keys:
            r = np.zeros_like(coarse_labels, dtype=bool)
            r[split_storage.splits[split_name].data] = True
            
            best_split_finder = ReturnBestSplit(experiment=experiment, splitter=Splitter(A=A, r=r), return_counterfactual_outputs=return_counterfactual_outputs(eligible_split_samples=r), score_thresh=score_thresh)
            
            try:
                split = best_split_finder.best_split(K, n_models, batch_starter_seed=0, in_memory=in_memory)
                IDs = split_storage.splits[split_name].data
                IDs_A = IDs[split[r]]
                IDs_B = IDs[np.invert(split[r])]
                split_storage.replace_split(split_name = split_name, new_split_1 = IDs_A, new_split_2 = IDs_B)
                
            except ValueError:
                split_storage.finish_split(split_name)

     
    return split_storage

    #TODO interface for clustering should be data in, get clusters (scikit learn)
    #TODO can test on a dataset where clusters are obvious (single features, other features encode subgroups - cluster 1 use dimension 3 for ground truth; run the whole thing from datamodels)
         # if it passes, can start running pipelines in the background (the interface shouldnt change)
         # while I clean up the code, improve it, and write more tests in the background