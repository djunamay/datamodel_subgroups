from ..counterfactuals.counterfactuals import CounterfactualEvaluation, CounterfactualExperimentResults
from ..counterfactuals.datahandling import PartitionStorageBase, CounterfactualInputs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering

def get_cfg_inputs(features, weights, features_filtered):

    n_features = weights.shape[1]

    # PCA regular
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_features) 
    pca_matrix = pca.fit_transform(X_scaled)

    # PCA filtered
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_filtered)
    pca = PCA(n_components=n_features) 
    pca_matrix_filtered = pca.fit_transform(X_scaled)

    return CounterfactualInputs(datamodel=weights, pca=pca_matrix, pca_filtered=pca_matrix_filtered)


def run_counterfactuals(cf_inputs, counterfactual_estimator, n_iter=100, n_clusters=2, random_state=1, shuffle_rng=None, model_rng=None):
    all_results = []
    for name, mat in cf_inputs:
        partition_storage = PartitionStorageBase(matrix=mat, partitioner=SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=random_state))
        experiment_result = CounterfactualExperimentResults(counterfactual_estimator=counterfactual_estimator, partition_storage=partition_storage, model_rng = model_rng, shuffle_rng = shuffle_rng, n_iter=n_iter)
        results = experiment_result.results
        results['input'] = str(name)
        all_results.append(results)
    return pd.concat(all_results)
