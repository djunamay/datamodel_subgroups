from .base import SplitFactoryInterface, SplitterArgsInterface, ReturnBestSplitInterface
from ..storage.base import Experiment
import chz
import numpy as np
from ..storage.results import ResultsStorage
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from numpy.typing import NDArray
from ..counterfactuals.outputs import ReturnCounterfactualOutputsBasic
from subgroups.storage.results import ResultsStorage
from sklearn.metrics.pairwise import cosine_similarity
from .base import ProcessExperimentForSplitsInterface


@chz.chz
class SpectralSplitWeights(SplitFactoryInterface): # TODO: this should be a function. Things should only be a class when it holds data.
    """
    Parameters
    ----------

    A:  NDArray of float, shape (n_samples, n_samples)
        Affinity matrix computed from datamodel weights or from other input

    r: NDarray of bool, shape (n_samples,)
        vector indicating which samples to split into two sets

    k: float
        percentile along embedding dimension at which to cut samples in r into two sets
    """

    A : NDArray[float]
    r : NDArray[bool]

    @chz.init_property
    def _A_r(self):
        "subset rows and columns by r"
        return self.A[self.r][:,self.r]

    @chz.init_property
    def _embedding(self):
        embedder = SpectralEmbedding(
                    n_components=1,
                    affinity='precomputed'
                )
        return embedder.fit_transform(self._A_r).reshape(-1)

    def _split(self, k, SplitterArgs):
        split = np.zeros(self.A.shape[0], dtype=bool)
        if SplitterArgs.greater_than:
            split[self.r] = self._embedding > np.percentile(self._embedding, k)  
        else:
            split[self.r] = self._embedding < np.percentile(self._embedding, k)  
        return split

    def split(self, k, SplitterArgs): 
        return self._split(k, SplitterArgs)

@chz.chz
class SpectralSplitWeightsArgs(SplitterArgsInterface):
    greater_than: bool

@chz.chz
class ReturnBestSpectralSplit(ReturnBestSplitInterface):
    experiment: Experiment
    splitter: SpectralSplitWeights
    return_counterfactual_outputs: ReturnCounterfactualOutputsBasic
    score_thresh: float

    def _return_scores(self, K, n_models, batch_starter_seed, in_memory):
        scores = []
        for state in [True, False]:
            s = self.get_true_scores_for_splits(experiment=self.experiment, splitter= self.splitter, return_counterfactual_outputs=self.return_counterfactual_outputs, SplitArgs=SpectralSplitWeightsArgs(greater_than=state),
            K=K, n_models=n_models, batch_starter_seed=batch_starter_seed, in_memory=in_memory)
            scores.append(s)
        return scores[0] + scores[1]

    def best_split(self, K, n_models, batch_starter_seed, in_memory):
        scores = self._return_scores(K, n_models, batch_starter_seed, in_memory)
        best_score = scores.mean(axis=1)
    
        if np.max(best_score) > self.score_thresh:
            return self.splitter.split(K[np.argmax(best_score)], SpectralSplitWeightsArgs(greater_than=True))
        else:
            raise ValueError


class CosineSim(ProcessExperimentForSplitsInterface):
    """A callable class that returns sample-wise scaled cosine similarities based on datamodels weights matrix"""

    def __call__(self, experiment) -> NDArray:
        res = ResultsStorage(experiment)
        cosine_sim = cosine_similarity(res.weights) 
        A = (cosine_sim+1)/2 
        return A

class CosineSimUntransformed(ProcessExperimentForSplitsInterface):
    """A callable class that returns sample-wise scaled cosine similarities based on untransformed features"""

    def __call__(self, experiment) -> NDArray:
        res = experiment.dataset.untransformed_features
        cosine_sim = cosine_similarity(res) 
        A = (cosine_sim+1)/2 
        return A

# @chz.chz
# class SpectralSplitWeights(SplitFactoryInterface):

#     experiment: Experiment
#     split_class_1: bool
#     percentage: float

#     @chz.init_property
#     def _results(self):
#         return ResultsStorage(self.experiment)

#     @chz.init_property
#     def _affinity_matrix(self):
#         cosine_sim = cosine_similarity(self._results.weights.T) 
#         return (cosine_sim+1)/2

#     @chz.init_property
#     def _embedding(self):
#         embedder = SpectralEmbedding(
#                     n_components=1,
#                     affinity='precomputed'
#                 )
#         return embedder.fit_transform(self._affinity_matrix).reshape(-1)

#     @chz.init_property
#     def _labels(self):
#         if self.split_class_1:
#             return self.experiment.dataset.coarse_labels
#         else: 
#             return np.invert(self.experiment.dataset.coarse_labels)

#     @chz.init_property
#     def _split(self):
#         split = np.zeros(len(self._labels), dtype=bool)
#         split[self._labels] = (self._embedding)[self._labels]>np.percentile(self._embedding[self._labels], self.percentage)  
#         return split

#     @chz.init_property
#     def split(self):
#         return self._split

# @chz.chz
# class PCASplitWeights(SplitFactoryInterface):

#     experiment: Experiment
#     split_class_1: bool
#     percentage: float
#     pc : int

#     @chz.init_property
#     def _results(self):
#         return ResultsStorage(self.experiment)

#     @chz.init_property
#     def _embedding(self):
#         embedder = PCA(n_components=self.pc+1, svd_solver="auto", random_state=0)
#         return embedder.fit_transform(self._results.weights.T)[:,self.pc].reshape(-1)

#     @chz.init_property
#     def _labels(self):
#         if self.split_class_1:
#             return self.experiment.dataset.coarse_labels
#         else: 
#             return np.invert(self.experiment.dataset.coarse_labels)

#     @chz.init_property
#     def _split(self):
#         split = np.zeros(len(self._labels), dtype=bool)
#         split[self._labels] = (self._embedding)[self._labels]>np.percentile(self._embedding[self._labels], self.percentage)  
#         return split

#     @chz.init_property
#     def split(self):
#         return self._split
    


# @chz.chz
# class PCASplitWeights2(SplitFactoryInterface):

#     experiment: Experiment
#     split_class_1: bool
#     percentage: float
#     pc : int

#     @chz.init_property
#     def _results(self):
#         return ResultsStorage(self.experiment)

#     @chz.init_property
#     def _embedding(self):
#         embedder = PCA(n_components=self.pc+1, svd_solver="auto", random_state=0)
#         return embedder.fit_transform(self._results.weights.T)[:,self.pc].reshape(-1)

#     @chz.init_property
#     def _labels(self):
#         if self.split_class_1:
#             return self.experiment.dataset.coarse_labels
#         else: 
#             return np.invert(self.experiment.dataset.coarse_labels)

#     @chz.init_property
#     def _split(self):
#         split = np.zeros(len(self._labels), dtype=bool)
#         split[self._labels] = (self._embedding)[self._labels]<np.percentile(self._embedding[self._labels], self.percentage)  
#         return split

#     @chz.init_property
#     def split(self):
#         return self._split

# @chz.chz
# class SpectralSplitFeatures(SplitFactoryInterface):

#     experiment: Experiment
#     split_class_1: bool
#     percentage: float

#     @chz.init_property
#     def _reduced_dim_features(self):
#         return self.experiment.dataset.features #self.experiment.dataset._reduce_dimensionality(self.experiment.dataset.untransformed_features, self.experiment.dataset.features.shape[0])

#     @chz.init_property
#     def _affinity_matrix(self):
#         cosine_sim = cosine_similarity(self._reduced_dim_features) 
#         return (cosine_sim+1)/2

#     @chz.init_property
#     def _embedding(self):
#         embedder = SpectralEmbedding(
#                     n_components=1,
#                     affinity='precomputed'
#                 )
#         return embedder.fit_transform(self._affinity_matrix).reshape(-1)

#     @chz.init_property
#     def _labels(self):
#         if self.split_class_1:
#             return self.experiment.dataset.coarse_labels
#         else: 
#             return np.invert(self.experiment.dataset.coarse_labels)

#     @chz.init_property
#     def _split(self):
#         split = np.zeros(len(self._labels), dtype=bool)
#         split[self._labels] = (self._embedding)[self._labels]>np.percentile(self._embedding[self._labels], self.percentage)  
#         return split

#     @chz.init_property
#     def split(self):
#         return self._split
    
# @chz.chz
# class PCASplitFeatures(SplitFactoryInterface):

#     experiment: Experiment
#     split_class_1: bool
#     percentage: float
#     pc: int

#     @chz.init_property
#     def _reduced_dim_features(self):
#         return self.experiment.dataset.features #self.experiment.dataset._reduce_dimensionality(self.experiment.dataset.untransformed_features, self.experiment.dataset.features.shape[0])


#     @chz.init_property
#     def _embedding(self):
#         return self._reduced_dim_features[:,self.pc]

#     @chz.init_property
#     def _labels(self):
#         if self.split_class_1:
#             return self.experiment.dataset.coarse_labels
#         else: 
#             return np.invert(self.experiment.dataset.coarse_labels)

#     @chz.init_property
#     def _split(self):
#         split = np.zeros(len(self._labels), dtype=bool)
#         split[self._labels] = (self._embedding)[self._labels]>np.percentile(self._embedding[self._labels], self.percentage)  
#         return split

#     @chz.init_property
#     def split(self):
#         return self._split