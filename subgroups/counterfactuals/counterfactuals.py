from ..datasamplers.base import RandomGeneratorTCInterface
from .datahandling import SplitClass, CoarseSplits
from sklearn.utils import shuffle
import chz
import numpy as np
from ..models.base import ModelFactory
from ..utils.scoring import compute_margins
from .base import CounterfactualEvaluationInterface, CounterfactualResultsInterface, SplitResultsInterface, PartitionStorageInterface
from ..utils.random import fork_rng
import pandas as pd
from itertools import product
from sklearn.metrics import roc_auc_score
from numpy.typing import NDArray

class SplitResults(SplitResultsInterface):

    def __init__(self, labels: np.ndarray[bool], probs_on_split: np.ndarray[float], probs_outside_split: np.ndarray[float]):
        self._labels = labels
        self._probs_on_split = probs_on_split
        self._probs_outside_split = probs_outside_split


    @property
    def labels(self)-> np.ndarray[bool]:
        return self._labels

    @property
    def probabilities_on_split(self)-> np.ndarray[float]:
        return self._probs_on_split

    @property
    def probabilities_outside_split(self)-> np.ndarray[float]:
        return self._probs_outside_split


class CounterfactualResults(CounterfactualResultsInterface):
    def __init__(self, test_labels: np.ndarray[bool], probs_split_a: np.ndarray[float], probs_split_b: np.ndarray[float], probs_outside_split_a: np.ndarray[float], probs_outside_split_b: np.ndarray[float]):
        self._test_labels = test_labels
        self._probs_split_a = probs_split_a
        self._probs_split_b = probs_split_b
        self._probs_outside_split_a = probs_outside_split_a
        self._probs_outside_split_b = probs_outside_split_b


    @property
    def split_a(self)-> SplitResultsInterface:
        return SplitResults(labels=self._test_labels, probs_on_split=self._probs_split_a, probs_outside_split=self._probs_outside_split_a)

    @property
    def split_b(self)-> SplitResultsInterface:
        return SplitResults(labels=self._test_labels, probs_on_split=self._probs_split_b, probs_outside_split=self._probs_outside_split_b)


@chz.chz
class CounterfactualEvaluation(CounterfactualEvaluationInterface):

    features: np.ndarray
    coarse_labels: np.ndarray
    train_size: int=None
    test_size: int=None
    train_fraction: float=None
    classifier: ModelFactory

    def _split_coarse(self, cluster_labels, sample_indices) -> CoarseSplits:
        if sample_indices is not None:
            return CoarseSplits(features=self.features[sample_indices], labels=self.coarse_labels[sample_indices], fine_label_bool=cluster_labels[sample_indices])
        else:
            return CoarseSplits(features=self.features, labels=self.coarse_labels, fine_label_bool=cluster_labels)

    def _make_group(self, split: SplitClass, seed: int, n_train: int, n_test: int) -> dict:
        X, y = shuffle(split.X, split.y, random_state=seed)
        return {
            'X_train': X[:n_train],
            'y_train': y[:n_train],
            'X_test':  X[n_train:n_train+n_test],
            'y_test':  y[n_train:n_train+n_test]
        }

    def _make_predictions(self, model, X_test):
        probabilities = model.predict_proba(X_test)
        return probabilities[:,1]
    
    
    def _prepare_data(self, cluster_labels, sample_indices, shuffle_rng):
        splits = self._split_coarse(cluster_labels=cluster_labels, sample_indices=sample_indices)
        smallest_group_size = np.min([splits.whole.X.shape[0], splits.split_a.X.shape[0], splits.split_b.X.shape[0]])
        if self.train_size is None:
            n_train = int(self.train_fraction * smallest_group_size)
            n_test = smallest_group_size - n_train
        else:
            n_train = self.train_size
            n_test = self.test_size
        train_data_shuffle_rngs_children = fork_rng(shuffle_rng, 3)
        return {
            'A': self._make_group(splits.whole, np.random.RandomState(train_data_shuffle_rngs_children[0].bit_generator), n_train, n_test),
            'B': self._make_group(splits.split_a, np.random.RandomState(train_data_shuffle_rngs_children[1].bit_generator), n_train, n_test),
            'C': self._make_group(splits.split_b, np.random.RandomState(train_data_shuffle_rngs_children[2].bit_generator), n_train, n_test),
        }
    
    def _counterfactual_evaluation(self, cluster_labels, sample_indices, model_rng, shuffle_rng):
        data = self._prepare_data(cluster_labels=cluster_labels, sample_indices=sample_indices, shuffle_rng=shuffle_rng)
        scenarios = {
            '1': ['A','B'],
            '2': ['A','C'],
        }
        results = {}
        for train_name, train_groups in scenarios.items():
            X_tr = np.vstack([data[g]['X_train'] for g in train_groups])
            y_tr = np.concatenate([data[g]['y_train'] for g in train_groups])
            model = self.classifier.build_model(rng=model_rng)
            model.fit(X_tr, y_tr)

            for test_name, test_groups in scenarios.items():
                X_te = np.vstack([data[g]['X_test'] for g in test_groups])
                y_te = np.concatenate([data[g]['y_test'] for g in test_groups])
                probs = self._make_predictions(model, X_te)
                results[f'train{train_name}_test{test_name}'] = {
                    'probs': probs
                }
        
        results['y_test'] = y_te

        return results
    
    def _counterfactual_evaluation_results(self, cluster_labels, sample_indices=None, model_rng=None, shuffle_rng=None):
        if len(np.unique(self.coarse_labels[cluster_labels]))!=1:
            raise ValueError("Cluster must be a subset of one of the coarse label classes.")
        
        results = self._counterfactual_evaluation(cluster_labels=cluster_labels, sample_indices=sample_indices, model_rng=model_rng, shuffle_rng=shuffle_rng)
        return CounterfactualResults(
            test_labels=results['y_test'],
            probs_split_a=results['train1_test1']['probs'],
            probs_split_b=results['train2_test2']['probs'],
            probs_outside_split_a=results['train2_test1']['probs'],
            probs_outside_split_b=results['train1_test2']['probs'],
        )
    
    def counterfactual_evaluation(self, partition, sample_indices=None, model_rng=None, shuffle_rng=None, n_iter=1000):
        split_names = ['split_a', 'split_b']
        prob_attrs = ['probabilities_on_split', 'probabilities_outside_split']

        all_outs = []

        build_model_rngs_children = fork_rng(model_rng, n_iter)
        train_data_shuffle_rngs_children = fork_rng(shuffle_rng, n_iter)

        for seed in range(n_iter):

            rows = []
            results = self._counterfactual_evaluation_results(partition, sample_indices=sample_indices, model_rng=build_model_rngs_children[seed], shuffle_rng=train_data_shuffle_rngs_children[seed])

            for split_name, prob_attr in product(split_names, prob_attrs):
                split = getattr(results, split_name)
                rows.append({
                    'split':     split_name,
                    'prob_type': prob_attr.replace("probabilities", "evaluation"),
                    'auc': roc_auc_score(split.labels, getattr(split, prob_attr)),
                    'mean_margins': np.mean(compute_margins(getattr(split, prob_attr), split.labels)),
                    'model_seed': seed
                })

            df = pd.DataFrame(rows)
            all_outs.append(df)
        
        return pd.concat(all_outs)
    

@chz.chz
class CounterfactualExperimentResults():
    
    partition_storage: PartitionStorageInterface
    counterfactual_estimator: CounterfactualEvaluationInterface
    n_iter: int
    model_rng: np.random.RandomState
    shuffle_rng: np.random.RandomState

    @staticmethod
    def _av_over_splits(res):
        avg_auc = (
            res
            .groupby(['model_seed', 'prob_type', 'subcluster_index'], as_index=False)[['auc', 'mean_margins']]
            .mean()
            .rename(columns={'auc': 'mean_auc', 'mean_margins': 'mean_margins'})
        )
        return avg_auc
    
    @staticmethod
    def _av_over_subclusters(res):
        avg_auc = (
            res
            .groupby(['model_seed', 'prob_type'], as_index=False)[['mean_auc', 'mean_margins']]
            .mean()
        )
        return avg_auc
    
    @chz.init_property
    def _partitions(self)->NDArray[int]:
        return self.partition_storage.partitions
    
    @chz.init_property
    def _counterfactual_experiment_results(self):
        all_outs = []
        for current_subcluster_index in range(self.partition_storage.n_partitions):
            all_outs.append(self._run_counterfactual_experiment(current_subcluster_index))
        return pd.concat(all_outs)


    def _partition_for_subcluster(self, current_subcluster_index: int)->NDArray[int]:
        
        if np.sum(self.counterfactual_estimator.coarse_labels)==len(self._partitions):
            group_to_subcluster_indices = self.counterfactual_estimator.coarse_labels
        else:
            group_to_subcluster_indices = np.invert(self.counterfactual_estimator.coarse_labels)

        partition = np.zeros_like(self.counterfactual_estimator.coarse_labels)
        partition[group_to_subcluster_indices] = self._partitions == current_subcluster_index

        return partition

    def _run_counterfactual_experiment(self, current_subcluster_index: int):
        partition = self._partition_for_subcluster(current_subcluster_index)
        results = self.counterfactual_estimator.counterfactual_evaluation(partition, model_rng=self.model_rng, shuffle_rng=self.shuffle_rng, n_iter=self.n_iter)
        results['subcluster_index'] = current_subcluster_index
        return results


    @chz.init_property
    def results(self):
        return self._av_over_subclusters(self._av_over_splits(self._counterfactual_experiment_results))
    