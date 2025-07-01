from .base import BaseDataset  
import numpy as np
import pandas as pd
import chz
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

@chz.chz
class RosmapSingleCellDataset(BaseDataset):
    """
    Processes the Rosmap Single Cell dataset, providing access to features, labels, and metadata.

    Attributes
    ----------
    """

    path_to_data: str = chz.field(doc="Path to the data")
    path_to_meta_data: str = chz.field(doc="Path to the meta data")
    path_to_extended_meta_data: str = chz.field(doc="Path to the extended meta data")
    path_to_projid_dictionary: str = chz.field(doc="Path to the projid dictionary")
    n_components: int = chz.field(default=50, doc="Number of components to use")

    @staticmethod
    def _get_stats(genes, funcs = [np.mean, np.std, lambda x, y: np.percentile(x, [10, 90], y).T]):
        curr_stats = []
        for func in funcs:
            result = func(genes, 0)
            if len(result.shape) == 1:
                result = result[:, None]
            curr_stats.append(result)
        return np.concatenate(curr_stats, 1).ravel()

    @chz.init_property
    def _full_single_cell_data(self):
        data = np.memmap(self.path_to_data, dtype='float32', shape = (2335726, 50), mode = 'r')
        return data[:, :self.n_components]

    @chz.init_property
    def _full_meta_data(self):
        return np.memmap(self.path_to_meta_data, dtype='uint16', shape = (2335726, 7), mode = 'r')

    @chz.init_property
    def _index(self):
        return self._full_meta_data[:,2]!=8
    
    @chz.init_property
    def _indexed_single_cell_data(self):
        return self._full_single_cell_data[self._index]
    
    @chz.init_property
    def _indexed_meta_data(self):
        return self._full_meta_data[self._index]
    
    @chz.init_property
    def _consecutive_patient_ids(self):
        patient_ids = self._indexed_meta_data[:,6] 
        patient_dict = dict(zip(np.unique(patient_ids), range(len(np.unique(patient_ids)))))
        return np.array([patient_dict[x] for x in patient_ids])
    
    @chz.init_property
    def _consecutive_celltype_ids(self):
        celltype = self._indexed_meta_data[:,2] 
        celltype_dict = dict(zip(np.unique(celltype), range(len(np.unique(celltype)))))
        return np.array([celltype_dict[x] for x in celltype])
    
    @chz.init_property
    def _order(self):
        return np.lexsort((self._consecutive_celltype_ids, self._consecutive_patient_ids)) # sort by patient ID, then by celltype
    
    @chz.init_property
    def _sorted_single_cell_data(self):
        return self._indexed_single_cell_data[self._order]
    
    @chz.init_property
    def _sorted_meta_data(self):
        return self._indexed_meta_data[self._order]
    
    @chz.init_property
    def _sorted_patient_ids(self):
        return self._consecutive_patient_ids[self._order]
    
    @chz.init_property
    def _sorted_celltype_ids(self):
        return self._consecutive_celltype_ids[self._order]

    @chz.init_property
    def _num_cell_types(self):
        return len(np.unique(self._sorted_celltype_ids))
    
    @chz.init_property
    def _num_samples(self):
        return len(np.unique(self._sorted_patient_ids))
    
    @chz.init_property 
    def _scaled_features(self):
         return np.vstack([StandardScaler().fit_transform(self._sorted_single_cell_data[self._sorted_celltype_ids==i]) for i in range(self._num_cell_types)])

    @chz.init_property
    def _starts_for_patient(self):
        starts = np.searchsorted(self._sorted_patient_ids, np.arange(self._sorted_patient_ids.max() + 1))
        return np.append(starts, len(self._sorted_patient_ids))
    
    @chz.init_property
    def _celltype_indices_per_patient(self):
        dict_for_patient = {}
        for i in range(self._num_samples):
            start, end = self._starts_for_patient[i:i+2]
            cells_for_patient = self._sorted_celltype_ids[start:end]
            starts = np.searchsorted(cells_for_patient, np.arange(cells_for_patient.max() + 1))+start
            dict_for_patient[i] = np.append(starts, len(cells_for_patient)+start)
        return dict_for_patient
    
    def _get_stats_for_patient(self, patient_id):
        stats_for_patient = []
        for celltype_id in range(self._num_cell_types):
            start, end = self._celltype_indices_per_patient[patient_id][celltype_id:celltype_id + 2]
            stats_for_patient.append(self._get_stats(self._scaled_features[start:end]))
        return np.concatenate(stats_for_patient)
    
    @chz.init_property
    def _features(self):
        out = [self._get_stats_for_patient(i) for i in range(self._num_samples)]
        return np.vstack(out)
    
    @chz.init_property
    def _projids(self):
        projid_dictionary = np.load(self.path_to_projid_dictionary)
        projid_dictionary = dict(zip(projid_dictionary[:,1], projid_dictionary[:,0]))
        labels = [self._sorted_meta_data[:,-1][x] for x in self._starts_for_patient[:-1]]
        return np.array([projid_dictionary[x] for x in labels])
    
    @chz.init_property
    def _extended_meta_data(self):
        full_metadata = pd.read_csv(self.path_to_extended_meta_data)
        full_metadata_sub = full_metadata.loc[full_metadata.groupby('projid')['fu_year'].idxmax()]
        full_metadata_sub.index = full_metadata_sub['projid']
        return full_metadata_sub.loc[self._projids]
    
    @property
    def features(self) -> NDArray[float]:
        return self._features
    
    @property
    def coarse_labels(self) -> NDArray[bool]:
        return np.array(self._extended_meta_data['AD_DLB_diagnosis']!=0)
    
    @property
    def fine_labels(self) -> NDArray[int]:
        return np.array(self._extended_meta_data['AD_DLB_diagnosis']).astype(int)

    @property 
    def descriptive_data(self) -> np.recarray:
        return self._extended_meta_data.to_records(index=False)
        