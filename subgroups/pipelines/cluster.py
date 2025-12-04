from sklearn.base import ClusterMixin, BaseEstimator, _fit_context
from sklearn.utils.validation import validate_data, check_random_state
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold._spectral_embedding import _spectral_embedding
from sklearn.utils._param_validation import (
    Hidden,
    Interval,
    MissingValues,
    Options,
    StrOptions,
    validate_params,
)
from numbers import Integral, Real
from subgroups.samplers.mask_generators import mask_factory_counterfactuals
from subgroups.pipelines.train_classifiers import run_training_batch
from tqdm import tqdm
import numpy as np
import chz
from functools import partial
from subgroups.storage.training import MaskMarginStorage
import ipdb
from itertools import chain
from dataclasses import dataclass
from numpy.typing import NDArray
from subgroups.storage.experiment import Experiment




# make this a dataclass
class CounterfactualClustering(ClusterMixin, BaseEstimator):
    _parameter_constraints: dict = {
        "eigen_solver": [StrOptions({"arpack", "lobpcg", "amg"}), None],
        "random_state": ["random_state"],
        "affinity": [
            callable,
            StrOptions(
               {"cosine_similarity", "precomputed"}
            ),
        ],
        "eigen_tol": [
            Interval(Real, 0.0, None, closed="left"),
            StrOptions({"auto"}),
        ],
    }
    def __init__(self, 
                experiment, 
                random_state = None, 
                eigen_solver = None, 
                eigen_tol = "auto", 
                affinity = 'cosine_similarity', 
                split_thresh = 1, 
                retrain = True, 
                retrain_batch_size = None, 
                retrain_alpha = None, 
                retrain_k = None, 
                verbose = True, 
                use_tqdm = True, 
                weights = None, 
                cluster_class_1 = True, 
                biases = None): 
                self.random_state = random_state
                self.eigen_tol = eigen_tol 
                self.affinity = affinity 
                self.split_thresh = split_thresh 
                self.eigen_solver = eigen_solver 
                self.retrain = retrain
                self.retrain_batch_size = retrain_batch_size
                self.retrain_alpha = retrain_alpha
                self.retrain_k = retrain_k 
                self.experiment = experiment 
                self.verbose = verbose 
                self.use_tqdm = use_tqdm 
                self.rng = np.random.default_rng(seed=self.random_state) 
                self.weights = weights 
                self.cluster_class_1 = cluster_class_1 
                self.biases = biases.reshape(-1,1)
                self.labels = self.experiment.dataset.coarse_labels if self.cluster_class_1 else ~self.experiment.dataset.coarse_labels 
                self.samples_per_class = int((len(self.labels) * self.retrain_alpha) / 2)
                self.min_test_fraction = 0.3

    def _split_score(self, retrain_output, split):
        """
        Compute difference in model accuracies on samples in split vs universe of samples outside of split belonging to the class being split.
        """
        masked_margins = np.ma.masked_array(retrain_output.margins, retrain_output.masks)
        #correct = (masked_margins>0) == retrain_output.labels
        logits = masked_margins/(2*retrain_output.labels-1) # TODO: why can't I get rid of this without the scores becoming negative in the last check at the end of the code?
        correct = (logits>0) == retrain_output.labels
        acc_in_split = np.sum(correct[:,split], axis=1)/np.sum(~correct[:,split].mask, axis=1)
        acc_out_split = np.sum(correct[:,self.labels & ~split], axis=1)/np.sum(~correct[:,self.labels & ~split].mask, axis=1)
        return np.mean(acc_in_split-acc_out_split)


    def _split_vectors(self, index, maps, k):
        """
        Return bool vector for current cluster indicating split along embedding.
        """
        split_extended_greater = np.zeros_like(index)
        split_extended_greater[index] = maps > np.percentile(maps, k)
        return split_extended_greater

    def _filter_splits(self, index, splits):
        keep = []
        for i in splits:
            if (i[index].sum()*(1-self.min_test_fraction) >= self.samples_per_class) and ((~i[index]).sum()*(1-self.min_test_fraction) >= self.samples_per_class):
                keep.append(i)
        return keep


    def _score_next_split(self, index):

        maps = _spectral_embedding(
            adjacency=self.affinity_matrix_[index][:,index],
            n_components=1,
            eigen_solver=self.eigen_solver,
            random_state=self.random_state,
            eigen_tol=self.eigen_tol
        ).reshape(-1)

        splits = self._filter_splits(index, [self._split_vectors(index, maps, k) for k in self.retrain_k])

        if len(splits) == 0:
            return 0, None

        if self.use_tqdm:
            it = tqdm(splits, desc='Evaluating next splits for this cluster', total=len(splits))
        else:
            it = splits

        scores = []
        all_splits = []

        for splits in it:
            split = [splits, (~splits & self.labels)]
            score = []
            for s in split:
                try:
                    mask_factory = partial(mask_factory_counterfactuals, alpha=self.retrain_alpha, split=s, min_test_fraction=self.min_test_fraction)
                    if self.retrain:
                        output = run_training_batch(self.experiment, batch_size=self.retrain_batch_size, mask_factory=mask_factory, use_tqdm=False, batch_starter_seed=self.rng.integers(0, 2 ** 32 - 1))
                    else:
                        output = MaskMarginStorage(n_models=self.retrain_batch_size, n_samples=self.experiment.dataset.num_samples, labels=self.labels, mask_factory=mask_factory,
                                        rng= self.experiment.tc_random_generator(batch_starter_seed=self.rng.integers(0, 2 ** 32 - 1)).mask_rng)
                        output.margins = (np.matmul(self.weights, output.masks.T)+self.biases).T

                    score.append(self._split_score(output, s))
            
                    
                except ValueError as e: # TODO fix these error messages
                    msg = str(e)
                    if "Cannot sample" in msg and "per class" in msg:
                        # sampling error: assign score to be zero here (don't have the power to make this split)
                        score = 0

                    elif "Bool split vector must index samples from one class only." in msg:
                        score = 0
                    
                    else:
                        # re-raise unknown ValueErrors
                        raise
            if (score[0] > self.split_thresh) & (score[1] > self.split_thresh):
                scores.append(np.max(score))
                all_splits.append(splits)

        if len(scores)>0:
            return scores[np.argmax(scores)], all_splits[np.argmax(scores)]
        else:
            return 0, None


    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        X = validate_data(
            self,
            X,
            accept_sparse=["csr", "csc", "coo"],
            dtype=np.float64,
            ensure_min_samples=2,
        )
        if not self.retrain and self.weights is None:
            self.weights = X
        if self.affinity == 'cosine_similarity':
            self.affinity_matrix_ = (cosine_similarity(X)+1)/2
        elif self.affinity == 'precomputed':
            self.affinity_matrix_ = X

        random_state = check_random_state(self.random_state)

        cluster_assignments = np.zeros(self.affinity_matrix_.shape[0])
        
        cluster_assignments[~self.labels] = -1
        working_clusters = set([0])
        done_clusters = set([-1])

        while working_clusters:
            current_clusters = list(working_clusters)
            for c in current_clusters:
                print(f'\033[4mTrying split for cluster {c}\033[0m')
                index = cluster_assignments == c
                score, split = self._score_next_split(index)
                if score > self.split_thresh:
                    print(index.sum())
                    print(split.sum())
                    new_cluster = max(cluster_assignments)+1
                    working_clusters.add(new_cluster)
                    cluster_assignments[split] = new_cluster
                    print(f"\033[3mSplit successful with score {score}, "
                    f"adding new cluster {new_cluster}.\033[0m")
                    #working_clusters = None
                elif score < self.split_thresh:
                    done_clusters.add(c)
                    working_clusters.remove(c)
                    print(f"\033[3mNo split happened with score {score}, "
                    f"moving cluster {c} to done clusters.\033[0m")

        self.labels_ = cluster_assignments

        scores = {}
        total_score = 0
        for i in np.unique(self.labels_)[1:]:
            print(i)
            s = self.labels_==i
            output = run_training_batch(self.experiment, batch_size=self.retrain_batch_size, mask_factory=partial(mask_factory_counterfactuals, alpha=self.retrain_alpha, split=s), use_tqdm=False, batch_starter_seed=self.rng.integers(0, 2 ** 32 - 1))
            score = self._split_score(output, s)
            scores[i] = score
            total_score +=score
        self.total_score = total_score
        self.scores = scores

        print(f"\033[1mDone splitting. A total of {len(np.unique(self.labels_))-1} clusters found.\033[0m")


