
from abc import abstractmethod, ABC
from numpy.typing import NDArray 
from functools import cached_property 
from dataclasses import dataclass 
from .experiment import Experiment
import glob
import numpy as np
from pathlib import Path
import re
from .base import ResultsStorageInterface

@dataclass
class ResultsStorage(ResultsStorageInterface):
    """
    Datamodel experiment results storage object
    """ 

    experiment: Experiment
    BATCH_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<batch>\d+)_(?P<suffix>weights|pearson_correlations|spearman_correlations|rmse|biases)$"
    )

    def __post_init__(self):
        """Run batch consistency check immediately upon object creation."""
        self._validate_batches()

    
    @staticmethod
    def _find_files_with_suffix(directory, suffix):
        """return file names including full path with a specific suffix within a specific directory"""
        pattern = f"{directory}/*{suffix}.npy"
        return sorted(glob.glob(pattern))

    def _validate_batches(self):
        """validate that filenames are complete and in same order"""
        file_names_dict = self._file_names_dict
        ref_batches = self._batch_number

        for key in file_names_dict.keys():
            current_batch_number = self._get_batch_number(key)
            if np.array_equal(ref_batches, current_batch_number):
                continue
            else:
                raise ValueError(f"Batch set mismatch for {key}.")

    def _load_1d_data(self, name):
        """Load specified data in order of sorted batches"""
        files_ordered = self._file_names_dict[name][self._batch_number_sorted_indices]
        return np.hstack([np.load(ix) for ix in files_ordered])

    def _get_batch_number(self, name):
        """Return array of batch numbers extracted from file names of specific data"""
        return np.array([int(self.BATCH_RE.match(Path(x).stem).groupdict()['batch']) for x in self._file_names_dict[name]])

    @cached_property
    def _file_names_dict(self):
        """dictionary of file names for the different outputs"""
        output = {}
        for i in ['weights', 'pearson_correlations', 'spearman_correlations', 'rmse', 'biases']:
            output[i] = np.array(self._find_files_with_suffix(self.experiment.path_to_datamodel_outputs, i))
        return output

    @cached_property
    def _batch_number(self):
        """return reference batch numbers"""
        return self._get_batch_number('weights')

    @cached_property
    def _batch_number_sorted_indices(self):
        """return indices for sorted reference batch numbers"""
        return np.argsort(self._batch_number)

    @cached_property
    def _weights(self):
        """return weights sorted by batch number"""
        files_ordered = self._file_names_dict['weights'][self._batch_number_sorted_indices]
        return np.vstack([np.load(ix) for ix in files_ordered])

    @cached_property
    def _pearson(self):
        """return pearson coefficients sorted by batch number"""
        return self._load_1d_data('pearson_correlations')

    @cached_property
    def _spearman(self):
        """return spearman coefficients sorted by batch number"""
        return self._load_1d_data('spearman_correlations')

    @cached_property
    def _bias(self):
        """return biases sorted by batch number"""
        return self._load_1d_data('biases')

    @cached_property
    def _rmse(self):
        """return rmse sorted by batch number"""
        return self._load_1d_data('rmse')

    @cached_property
    def _sample_indices(self):
        """return sample indices that correspond to the order of rows in weights"""
        sorted_batch_numbers = self._batch_number[self._batch_number_sorted_indices]
        return np.array([self.experiment.indices_to_fit(x) for x in sorted_batch_numbers]).reshape(-1)
    
    @cached_property
    def _rows_to_keep(self):
        return self._rmse!=0

    @cached_property
    def weights(self) -> NDArray: 
        """datamodel experiment weights (shape: [n_samples, n_samples] stacked in batch order)""" 
        return self._weights[self._rows_to_keep]

    @cached_property
    def pearson(self) -> NDArray: 
        """datamodel experiment pearson (shape: [n_samples] stacked in batch order)""" 
        return self._pearson[self._rows_to_keep]

    @cached_property
    def spearman(self) -> NDArray: 
        """datamodel experiment spearman (shape: [n_samples] stacked in batch order)""" 
        return self._pearson[self._rows_to_keep]
    
    @cached_property
    def bias(self) -> NDArray: 
        """datamodel experiment bias (shape: [n_samples] stacked in batch order)""" 
        return self._pearson[self._rows_to_keep]

    @cached_property
    def rmse(self) -> NDArray: 
        """datamodel experiment rmse (shape: [n_samples] stacked in batch order)""" 
        return self._pearson[self._rows_to_keep]
    
    @cached_property
    def sample_indices(self) -> NDArray: 
        """datamodel experiment sample_indices (shape: [n_samples] stacked in batch order)""" 
        return self._pearson[self._rows_to_keep]
        
