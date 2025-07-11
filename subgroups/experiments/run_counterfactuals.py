from ..counterfactuals.counterfactuals import CounterfactualEvaluation, CounterfactualExperimentResults
from ..counterfactuals.datahandling import PartitionStorageBase
import pandas as pd
from sklearn.cluster import SpectralClustering


def run_counterfactuals(cf_inputs, counterfactual_estimator, n_iter=100, n_clusters=2, random_state=None, shuffle_rng=None, model_rng=None):
    all_results = []
    for name, mat in cf_inputs:
        partition_storage = PartitionStorageBase(matrix=mat, partitioner=SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=random_state))
        experiment_result = CounterfactualExperimentResults(counterfactual_estimator=counterfactual_estimator, partition_storage=partition_storage, model_rng = model_rng, shuffle_rng = shuffle_rng, n_iter=n_iter)
        results = experiment_result.results
        results['input'] = str(name)
        all_results.append(results)
    return pd.concat(all_results)
