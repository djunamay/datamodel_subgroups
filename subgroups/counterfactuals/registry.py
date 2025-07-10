from .datahandling import CounterfactualInputsGTExSubset
from .counterfactuals import CounterfactualEvaluation
from ..datasamplers.feature_selectors import SelectPCsBasic
from ..models.classifier import XgbFactory
import numpy as np
from ..datastorage.registry import gtex_subset_experiment_home

#### GTEx subset experiment ####
def gtex_counterfactual_inputs():
    experiment = gtex_subset_experiment_home()
    return CounterfactualInputsGTExSubset(path_to_features="/Users/djuna/Documents/subgroups_data/gtex_subset/subset_esophagus_bloodvessel.gct", 
                                                       path_to_weights="/Users/djuna/Documents/temp/results/gtex_subset_experiment_june_30/datamodel_outputs/", 
                                                       dataset=experiment.dataset,
                                                       group_1=False)

def gtex_counterfactual_estimator():
    experiment = gtex_subset_experiment_home()
    return CounterfactualEvaluation(features=experiment.dataset.features[:,SelectPCsBasic().feature_indices(n_pcs=20)],
                                                        coarse_labels=experiment.dataset.coarse_labels,
                                                        train_size=int((0.012507530044163674*experiment.dataset.num_samples)/2), # corresponds to approx original model alpha of 0.012507530044163674
                                                        test_size=int((0.1*experiment.dataset.num_samples)/2),
                                                        classifier=XgbFactory(max_depth=7),
                                                        group_1=False)

####