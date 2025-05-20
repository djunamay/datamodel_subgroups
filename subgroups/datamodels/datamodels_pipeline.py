from .base import DatamodelsPipelineInterface
from .regressor import DataModelFactory

import numpy as np
import os
import chz
from sklearn.utils import shuffle
from tqdm import tqdm
from scipy.stats import pearsonr
from typing import Optional, Union
from numpy.typing import NDArray
import glob
from sklearn.metrics import root_mean_squared_error
Array = Union[np.ndarray, np.memmap]

@chz.chz
class DatamodelsPipelineBasic(DatamodelsPipelineInterface):
    path_to_inputs: str = chz.field()
    datamodel_factory: DataModelFactory = chz.field()

    @staticmethod
    def find_files(directory, search_pattern):
        pattern = f"{directory}/*{search_pattern}.npy"
        return sorted(glob.glob(pattern))

    @staticmethod
    def stack_memmap_files(in_paths, out_path):
        srcs  = [np.lib.format.open_memmap(p, mode="r") for p in in_paths]
        ref_dtype = srcs[0].dtype
        total =np.sum([i.shape[0] for i in srcs])
        out_shape = (int(total), srcs[0].shape[1])
        out = np.lib.format.open_memmap(out_path,
                                        mode="w+",
                                        dtype=ref_dtype,
                                        shape=out_shape)   
        offset = 0
        for arr in srcs:
            n = arr.shape[0]
            out[offset:offset+n] = arr.copy()    
            offset += n

        out._mmap.close() 

    @chz.init_property
    def _mask_input_paths(self):
        return self.find_files(self.path_to_inputs, "masks")

    @chz.init_property
    def _margins_input_paths(self):
        return self.find_files(self.path_to_inputs, "margins")

    @chz.init_property
    def _batch_order_masks(self):
        return np.array([int(x.split('_')[-2]) for x in self._mask_input_paths])

    @chz.init_property
    def _batch_order_margins(self):
        return np.array([int(x.split('_')[-2]) for x in self._margins_input_paths])

    @chz.init_property
    def _masks(self):
        out_path = os.path.join(self.path_to_inputs, "masks_concatenated.npy")
        
        if os.path.exists(out_path):
            return np.load(out_path, mmap_mode="r") #if self.n_test is None else np.load(out_path, mmap_mode="r")[:self.n_train+self.n_test]
        
        elif np.array_equal(self._batch_order_masks, self._batch_order_margins):
            self.stack_memmap_files(self._mask_input_paths, out_path)
            return np.load(out_path, mmap_mode="r") #if self.n_test is None else np.load(out_path, mmap_mode="r")[:self.n_train+self.n_test]
        
        else:
            raise ValueError("The number of batches and/or their order are not the same for masks and margins")
        
    @chz.init_property
    def _margins(self):
        out_path = os.path.join(self.path_to_inputs, "margins_concatenated.npy")

        if os.path.exists(out_path):
            return np.load(out_path, mmap_mode="r") #if self.n_test is None else np.load(out_path, mmap_mode="r")[:self.n_train+self.n_test]
        
        elif np.array_equal(self._batch_order_masks, self._batch_order_margins):
            self.stack_memmap_files(self._margins_input_paths, out_path)
            return np.load(out_path, mmap_mode="r") #if self.n_test is None else np.load(out_path, mmap_mode="r")[:self.n_train+self.n_test]
        
        else:
            raise ValueError("The number of batches and/or their order are not the same for masks and margins")
    
    @staticmethod
    def _create_array(in_memory: bool, path: Optional[str], dtype: np.dtype, shape: tuple[int, int]) -> Array:
        """
        Create an array either in memory or as a memory-mapped file.

        Parameters
        ----------
        in_memory : bool
            Flag to determine if the array is stored in memory.
        path : Optional[str]
            Path for memory-mapped file if not in memory.
        dtype : np.dtype
            Data type of the array.
        shape : tuple[int, int]
            Shape of the array.

        Returns
        -------
        Array
            Initialized array.
        """
        if in_memory:
            return np.zeros(shape, dtype=dtype)

        if path is None:
            raise ValueError("path must be provided when in_memory=False")

        mode = "r+" if os.path.exists(path) else "w+"
        return np.lib.format.open_memmap(path, dtype=dtype, mode=mode, shape=shape)

    def _get_samples_for_model(self, sample_index: int):
        index = np.invert(self._masks[:,sample_index])
        y = self._margins[index, sample_index]
        X = self._masks[index]
        return X, y
    
    def _fit_one_model(self, sample_index: int, seed: int, n_train: int, n_test: int):
        X, y = self._get_samples_for_model(sample_index)
        X, y = shuffle(X, y, random_state=seed)
        model = self.datamodel_factory.build_model(seed=seed)
        model.fit(X[:n_train], y[:n_train])
        start = 0

        if n_test is not None:
            stop = start + n_test
        else:
            stop = X.shape[0]

        y_hat = model.predict(X[start:stop])
        correlation = pearsonr(y[start:stop], y_hat)[0]
        rmse = root_mean_squared_error(y[start:stop], y_hat)
        
        return {'weights': model.coef_, 
                'bias': model.intercept_, 
                'correlation': correlation,
                'rmse': rmse}
    

    def fit_datamodels(self, indices, rng, n_train, n_test: int=None, path_to_outputs: str=None, in_memory: bool=True):

        weights = self._create_array(in_memory, None if in_memory else path_to_outputs + "_weights.npy",
            np.float32, (len(indices), self._masks.shape[1])
        )
        biases = self._create_array(in_memory, None if in_memory else path_to_outputs + "_biases.npy",
            np.float32, (len(indices),)
        )
        correlations = self._create_array(in_memory, None if in_memory else path_to_outputs + "_correlations.npy",
            np.float32, (len(indices),)
        )
        rmse = self._create_array(in_memory, None if in_memory else path_to_outputs + "_rmse.npy",
            np.float32, (len(indices),)
        )
        for i, sample_index in tqdm(enumerate(indices), total=len(indices)):
            if not correlations[i]==0:
                continue
            model = self._fit_one_model(sample_index, rng.integers(0, 2**32 - 1), n_train, n_test)
            weights[i] = model['weights']
            biases[i] = model['bias']
            correlations[i] = model['correlation']
            rmse[i] = model['rmse']

        if in_memory:
            return {'weights': weights, 
                    'biases': biases, 
                    'correlations': correlations,
                    'rmse': rmse}
        else:
            return path_to_outputs

        
