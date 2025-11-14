from .base import SplitFactoryInterface
from ..datastorage.base import Experiment
import chz
import numpy as np
from ..datastorage.results import ResultsStorage
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from numpy.typing import NDArray

@chz.chz
class SpectralSplitWeights(SplitFactoryInterface):
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
    greater_than : bool

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

    def _split(self, k):
        split = np.zeros(self.A.shape[0], dtype=bool)
        if self.greater_than:
            split[self.r] = self._embedding > np.percentile(self._embedding, k)  
        else:
            split[self.r] = self._embedding < np.percentile(self._embedding, k)  
        return split

    def split(self, k):
        return self._split(k)

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

@chz.chz
class PCASplitWeights(SplitFactoryInterface):

    experiment: Experiment
    split_class_1: bool
    percentage: float
    pc : int

    @chz.init_property
    def _results(self):
        return ResultsStorage(self.experiment)

    @chz.init_property
    def _embedding(self):
        embedder = PCA(n_components=self.pc+1, svd_solver="auto", random_state=0)
        return embedder.fit_transform(self._results.weights.T)[:,self.pc].reshape(-1)

    @chz.init_property
    def _labels(self):
        if self.split_class_1:
            return self.experiment.dataset.coarse_labels
        else: 
            return np.invert(self.experiment.dataset.coarse_labels)

    @chz.init_property
    def _split(self):
        split = np.zeros(len(self._labels), dtype=bool)
        split[self._labels] = (self._embedding)[self._labels]>np.percentile(self._embedding[self._labels], self.percentage)  
        return split

    @chz.init_property
    def split(self):
        return self._split
    


@chz.chz
class PCASplitWeights2(SplitFactoryInterface):

    experiment: Experiment
    split_class_1: bool
    percentage: float
    pc : int

    @chz.init_property
    def _results(self):
        return ResultsStorage(self.experiment)

    @chz.init_property
    def _embedding(self):
        embedder = PCA(n_components=self.pc+1, svd_solver="auto", random_state=0)
        return embedder.fit_transform(self._results.weights.T)[:,self.pc].reshape(-1)

    @chz.init_property
    def _labels(self):
        if self.split_class_1:
            return self.experiment.dataset.coarse_labels
        else: 
            return np.invert(self.experiment.dataset.coarse_labels)

    @chz.init_property
    def _split(self):
        split = np.zeros(len(self._labels), dtype=bool)
        split[self._labels] = (self._embedding)[self._labels]<np.percentile(self._embedding[self._labels], self.percentage)  
        return split

    @chz.init_property
    def split(self):
        return self._split

@chz.chz
class SpectralSplitFeatures(SplitFactoryInterface):

    experiment: Experiment
    split_class_1: bool
    percentage: float

    @chz.init_property
    def _reduced_dim_features(self):
        return self.experiment.dataset.features #self.experiment.dataset._reduce_dimensionality(self.experiment.dataset.untransformed_features, self.experiment.dataset.features.shape[0])

    @chz.init_property
    def _affinity_matrix(self):
        cosine_sim = cosine_similarity(self._reduced_dim_features) 
        return (cosine_sim+1)/2

    @chz.init_property
    def _embedding(self):
        embedder = SpectralEmbedding(
                    n_components=1,
                    affinity='precomputed'
                )
        return embedder.fit_transform(self._affinity_matrix).reshape(-1)

    @chz.init_property
    def _labels(self):
        if self.split_class_1:
            return self.experiment.dataset.coarse_labels
        else: 
            return np.invert(self.experiment.dataset.coarse_labels)

    @chz.init_property
    def _split(self):
        split = np.zeros(len(self._labels), dtype=bool)
        split[self._labels] = (self._embedding)[self._labels]>np.percentile(self._embedding[self._labels], self.percentage)  
        return split

    @chz.init_property
    def split(self):
        return self._split
    
@chz.chz
class PCASplitFeatures(SplitFactoryInterface):

    experiment: Experiment
    split_class_1: bool
    percentage: float
    pc: int

    @chz.init_property
    def _reduced_dim_features(self):
        return self.experiment.dataset.features #self.experiment.dataset._reduce_dimensionality(self.experiment.dataset.untransformed_features, self.experiment.dataset.features.shape[0])


    @chz.init_property
    def _embedding(self):
        return self._reduced_dim_features[:,self.pc]

    @chz.init_property
    def _labels(self):
        if self.split_class_1:
            return self.experiment.dataset.coarse_labels
        else: 
            return np.invert(self.experiment.dataset.coarse_labels)

    @chz.init_property
    def _split(self):
        split = np.zeros(len(self._labels), dtype=bool)
        split[self._labels] = (self._embedding)[self._labels]>np.percentile(self._embedding[self._labels], self.percentage)  
        return split

    @chz.init_property
    def split(self):
        return self._split