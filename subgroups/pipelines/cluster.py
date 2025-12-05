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
                cluster_class_1 = True, 
                weights = None,
                biases = None,
                validate = True): 
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
                self.cluster_class_1 = cluster_class_1 
                self.biases = biases.reshape(-1,1)
                self.labels = self.experiment.dataset.coarse_labels if self.cluster_class_1 else ~self.experiment.dataset.coarse_labels 
                self.samples_per_class = int((len(self.labels) * self.retrain_alpha) / 2)
                self.min_test_fraction = 0.3
                self.max_train_fraction = 1-self.min_test_fraction
                self.weights = weights
                self.validate = validate

    def _split_score(self, retrain_output, split):
        """
        Compute difference in model accuracies on samples in split vs universe of samples outside of split belonging to the class being split.
        """
        masked_margins = np.ma.masked_array(retrain_output.margins, retrain_output.masks)
        correct = masked_margins > 0

        correct_inside_split = correct[:,split]  
        correct_outside_split = correct[:,self.labels & ~split]

        acc_in_split = np.sum(correct_inside_split, axis=1)/np.sum(~correct_inside_split.mask, axis=1)
        acc_out_split = np.sum(correct_outside_split, axis=1)/np.sum(~correct_outside_split.mask, axis=1)

        return np.mean(acc_in_split-acc_out_split)


    def _split_vectors(self, index, maps, k):
        """
        Return bool vector for current cluster indicating split along embedding.
        """
        split = np.zeros_like(index)
        split[index] = maps > np.percentile(maps, k)
        return split

    def _score_next_split(self, index):

        maps = _spectral_embedding(
            adjacency=self.affinity_matrix_[index][:,index],
            n_components=1,
            eigen_solver=self.eigen_solver,
            random_state=self.random_state,
            eigen_tol=self.eigen_tol
        ).reshape(-1)

        # any k's that results in a cluster too small for training must be removed
        split_size = index.sum()
        valid_split = np.array([int(min(split_size-k*split_size, k*split_size)*self.max_train_fraction) for k in self.retrain_k/100]) > self.samples_per_class
        
        if valid_split.sum()==0:
            # return score of 0 if none of these splits are valid and cluster must be retired
            return 0, None

        else: 
            split_vectors = [self._split_vectors(index, maps, k) for k in self.retrain_k[valid_split]]

        
        best_split_score = 0
        best_split_vector = None

        for vector in tqdm(split_vectors, desc='Evaluating next splits for this cluster', total=len(split_vectors)) if self.use_tqdm else split_vectors:
            
            temporary_split_scores = []
            
            for s in [vector, (~vector & self.labels)]:

                # for both the left and the right hand side of the split vector, compute the counterfactual or actual (retrain is True) split score
                
                mask_factory = partial(mask_factory_counterfactuals, alpha=self.retrain_alpha, split=s, min_test_fraction=self.min_test_fraction)
                
                if self.retrain:
                    output = run_training_batch(self.experiment, 
                                                batch_size=self.retrain_batch_size, 
                                                mask_factory=mask_factory, 
                                                use_tqdm=False, 
                                                batch_starter_seed=self.rng.integers(0, 2 ** 32 - 1))
                else:
                    output = MaskMarginStorage(n_models=self.retrain_batch_size, 
                                               n_samples=self.experiment.dataset.num_samples, 
                                               labels=self.labels, 
                                               mask_factory=mask_factory,
                                               rng= self.experiment.tc_random_generator(batch_starter_seed=self.rng.integers(0, 2 ** 32 - 1)).mask_rng)

                    output.margins = (np.matmul(self.weights, output.masks.T)+self.biases).T

                temporary_split_scores.append(self._split_score(output, s))
    
            if (np.all(np.array(temporary_split_scores) > self.split_thresh)) and (np.max(temporary_split_scores) > best_split_score):
                # Only if a split results in two clusters that BOTH pass the split threshold AND if one of the scores is greater than the current best score, do we keep this split as an option 
                best_split_score = np.max(temporary_split_scores) # keep the best score as a metric for this split
                best_split_vector = vector

        return best_split_score, best_split_vector
   


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
            print("'weights' attribute is None. X is assumed to be the datamodels weights matrix.")
            self.weights = X
        if self.affinity == 'cosine_similarity':
            self.affinity_matrix_ = (cosine_similarity(X)+1)/2
        elif self.affinity == 'precomputed':
            self.affinity_matrix_ = X

        random_state = check_random_state(self.random_state)

        cluster_assignments = np.zeros(self.affinity_matrix_.shape[0])
        
        # samples from the opposite class (i.e. not up for splitting) are assigned a final label of -1
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

                    new_cluster = max(cluster_assignments)+1
                    working_clusters.add(new_cluster)
                    cluster_assignments[split] = new_cluster

                    print(f"\033[3mSplit successful with score {score}, "
                    f"adding new cluster {new_cluster}.\033[0m")

                elif score < self.split_thresh:

                    done_clusters.add(c)
                    working_clusters.remove(c)

                    print(f"\033[3mNo split happened with score {score}, "
                    f"moving cluster {c} to done clusters.\033[0m")

        self.labels_ = cluster_assignments
        self.unique_labels_ = np.unique(self.labels_)[1:]

        if self.validate:
            self.label_scores_ = {}
            self.total_score_ = 0

            for i in self.unique_labels_:
                s = self.labels_==i
                output = run_training_batch(self.experiment, batch_size=self.retrain_batch_size, mask_factory=partial(mask_factory_counterfactuals, alpha=self.retrain_alpha, split=s), use_tqdm=False, batch_starter_seed=self.rng.integers(0, 2 ** 32 - 1))
                score = self._split_score(output, s)
                self.label_scores_[i] = score
                self.total_score_ += score
           
        print(f"\033[1mDone splitting. A total of {len(np.unique(self.labels_))-1} clusters found.\033[0m")


