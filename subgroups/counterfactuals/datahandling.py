import numpy as np
from numpy.typing import NDArray
from ..models.base import SklearnClassifier
import chz
from .base import PartitionStorageInterface, CounterfactualInputsInterface, CounterfactualInputs
import numpy as np
import pandas as pd
from ..utils.loading import load_weights_data, load_eval_data
from ..datasets.base import DatasetInterface

class SplitClass:

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def X(self) -> np.ndarray:
        return self.X   

    def y(self) -> np.ndarray:
        return self.y

class CoarseSplits:

    def __init__(self, features: np.ndarray[float], labels: np.ndarray[bool], fine_label_bool: np.ndarray[bool]):
        self.features = features
        self.fine_label_bool = fine_label_bool
        self.labels = labels if len(np.unique(self.fine_label_bool[labels]))>1 else ~labels

    @property
    def whole(self) -> SplitClass:
        return SplitClass(X = self.features[~self.fine_label_bool & ~self.labels], y = self.labels[~self.fine_label_bool & ~self.labels])

    @property
    def split_a(self) -> SplitClass:
        return SplitClass(X = self.features[~self.fine_label_bool & self.labels], y = self.labels[~self.fine_label_bool & self.labels])

    @property
    def split_b(self) -> SplitClass:
        return SplitClass(X = self.features[self.fine_label_bool], y = self.labels[self.fine_label_bool])


@chz.chz
class PartitionStorageBase(PartitionStorageInterface):

    matrix: NDArray[float]
    partitioner: SklearnClassifier

    @chz.init_property
    def _correlation_matrix(self):
        corr = np.corrcoef(self.matrix)
        return (corr+1)/2
    
    @chz.init_property
    def partitions(self)-> NDArray[int]:
        return self.partitioner.fit_predict(self._correlation_matrix)
    
    @chz.init_property
    def n_partitions(self)-> int:
        return len(np.unique(self.partitions))


@chz.chz
class CounterfactualInputsBasic(CounterfactualInputsInterface):
    
    path_to_features: str
    path_to_weights: str
    dataset: DatasetInterface
    group_1: bool

    @chz.init_property
    def sample_index(self)->np.ndarray:
        return self.dataset.coarse_labels if self.group_1 else np.invert(self.dataset.coarse_labels)

    @chz.init_property
    def _features(self)->np.ndarray:
        tpm_data = pd.read_csv(self.path_to_features, sep='\t', skiprows=2)
        return tpm_data.iloc[:, 2:].T


    @chz.init_property
    def _features_filtered(self)->np.ndarray:

        tmp_data_low_removed = self._features.loc[:,(self._features==0).sum(axis=0)<(0.3*self._features.shape[0])]
        avs = tmp_data_low_removed.groupby(self.sample_index).mean()
        avs_array = np.array(avs) + np.finfo(float).eps
        lfcs = np.array(np.abs(np.log2(avs_array[0]/avs_array[1])))
        features_subset = self._features.iloc[:,np.argsort(lfcs)[-np.sum(self.sample_index):]]
        
        return features_subset.loc[self.sample_index].values.astype(float)

    @chz.init_property
    def _pca_input(self)->np.ndarray:
        
        return self._features.loc[self.sample_index].values.astype(float)
    
    @chz.init_property
    def _datamodel_input(self)->np.ndarray:
        weights, _ = load_weights_data(self.path_to_weights)
        x = load_eval_data(self.path_to_weights, 'pearson_correlations')[0]
        weights = weights[x!=0]
        return weights[self.sample_index][:,self.sample_index]
    
    @chz.init_property
    def _pca_filtered_input(self)->np.ndarray:

        return self._features_filtered
    
    @chz.init_property
    def to_counterfactual_inputs(self) -> CounterfactualInputs:
        return CounterfactualInputs(pca_input=self._pca_input, datamodel_input=self._datamodel_input, pca_filtered_input=self._pca_filtered_input)