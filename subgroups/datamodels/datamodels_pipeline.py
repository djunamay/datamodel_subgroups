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
    """
    A basic implementation of DatamodelsPipelineInterface, which fits a SklearnRegressor to each sample specified in the indices.
    The datamodels are fitted independently of one another.
    """
    path_to_inputs: str = chz.field(doc='Path to classifier outputs containing the masks and margins.')
    datamodel_factory: DataModelFactory = chz.field(doc='Factory for creating the datamodel.')
    path_to_outputs: str = chz.field(default=None, doc='Path to save the datamodel outputs.')

    @staticmethod
    def _find_files(directory, search_pattern):
        """
        Find all files in the given directory that match the search pattern.

        Parameters
        ----------
        directory : str
            Directory to search for files.
        search_pattern : str
            Pattern to search for in the filenames.

        Returns
        -------
        List[str]
            List of file paths that match the search pattern.
        """
        pattern = f"{directory}/*{search_pattern}.npy"
        return sorted(glob.glob(pattern))

    @chz.init_property
    def _model_completed_indices(self):
        out_path = os.path.join(self.path_to_inputs, "masks_concatenated.npy")
        
        if os.path.exists(out_path):
            return None
        else:
            in_paths = self._find_files(self.path_to_inputs, 'test_accuracies')
            srcs  = [np.lib.format.open_memmap(p, mode="r") for p in in_paths]
            return [x!=0 for x in srcs]
        
    def _stack_memmap_files(self, in_paths, out_path):
        """
        Stack multiple memory-mapped files into a single file.

        Parameters
        ----------
        in_paths : List[str]
            List of file paths to the memory-mapped files to be stacked.
        out_path : str
            Path to save the stacked memory-mapped file.

        Returns
        -------
        None
        """
        srcs  = [np.lib.format.open_memmap(p, mode="r") for p in in_paths]
        ref_dtype = srcs[0].dtype
        total = np.sum(np.array([x.sum() for x in self._model_completed_indices]))
        out_shape = (int(total), srcs[0].shape[1])
        out = np.lib.format.open_memmap(out_path,
                                        mode="w+",
                                        dtype=ref_dtype,
                                        shape=out_shape)   
        offset = 0
        for i, arr in enumerate(srcs):
            arr = arr[self._model_completed_indices[i]]
            n = arr.shape[0]
            out[offset:offset+n] = arr.copy()    
            offset += n

        out._mmap.close() 

    @property
    def _mask_input_paths(self):
        """
        Get the file paths for the mask files.
        """
        return self._find_files(self.path_to_inputs, "masks")

    @property
    def _margins_input_paths(self):
        """
        Get the file paths for the margin files.
        """
        return self._find_files(self.path_to_inputs, "margins")

    @property
    def _batch_order_masks(self):
        """
        Get the batch order for the mask files.
        """
        return np.array([int(x.split('_')[-2]) for x in self._mask_input_paths])

    @property
    def _batch_order_margins(self):
        """
        Get the batch order for the margin files.
        """
        return np.array([int(x.split('_')[-2]) for x in self._margins_input_paths])

    @property
    def _masks(self):
        """
        Access the concatenated mask files.
        Verifies that the number of batches and their order - as used for concatenation - are the same for masks and margins.
        """
        out_path = os.path.join(self.path_to_inputs, "masks_concatenated.npy")
        
        if os.path.exists(out_path):
            return np.load(out_path, mmap_mode="r") #if self.n_test is None else np.load(out_path, mmap_mode="r")[:self.n_train+self.n_test]
        
        elif np.array_equal(self._batch_order_masks, self._batch_order_margins):
            mask_input_paths = self._mask_input_paths
            if len(mask_input_paths) == 0:
                raise ValueError("No mask files found. Did you run the training pipeline?")
            self._stack_memmap_files(in_paths=mask_input_paths, out_path=out_path)
            return np.load(out_path, mmap_mode="r") #if self.n_test is None else np.load(out_path, mmap_mode="r")[:self.n_train+self.n_test]
        
        else:
            raise ValueError("The number of batches and/or their order are not the same for masks and margins")
        
    @property
    def _margins(self):
        """
        Access the concatenated margin files.
        Verifies that the number of batches and their order - as used for concatenation - are the same for masks and margins.
        """
        out_path = os.path.join(self.path_to_inputs, "margins_concatenated.npy")

        if os.path.exists(out_path):
            return np.load(out_path, mmap_mode="r") #if self.n_test is None else np.load(out_path, mmap_mode="r")[:self.n_train+self.n_test]
        
        elif np.array_equal(self._batch_order_masks, self._batch_order_margins):
            margin_input_paths = self._margins_input_paths
            if len(margin_input_paths) == 0:
                raise ValueError("No margin files found. Did you run the training pipeline?")
            self._stack_memmap_files(in_paths=margin_input_paths, out_path=out_path)
            return np.load(out_path, mmap_mode="r") #if self.n_test is None else np.load(out_path, mmap_mode="r")[:self.n_train+self.n_test]
        
        else:
            raise ValueError("The number of batches and/or their order are not the same for masks and margins")
    
    @staticmethod
    def _rng(seed):
        """
        Generate two independent random number generators.
        """
        seq = np.random.SeedSequence(seed)
        rngs_fit = np.random.default_rng(seq.spawn(1)[0])
        rngs_shuffle = np.random.default_rng(seq.spawn(1)[0])
        return rngs_fit, rngs_shuffle

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

    @staticmethod
    def _get_samples_for_model(masks, margins, sample_index: int):
        """
        Get the samples for the model.
        For a given sample, exclude any models for fitting in which that sample was included in the training set.
        """
        index = np.invert(masks[:,sample_index])
        y = margins[index, sample_index]
        X = masks[index]
        return X, y
    
    @staticmethod
    def _train_samples(X, y, n_train):
        return X[:n_train], y[:n_train]
    
    @staticmethod
    def _test_samples(X, y, n_train, n_test):
        start = n_train
        if n_test is not None:
            stop = start + n_test
        else:
            stop = X.shape[0]
        return X[start:stop], y[start:stop]
    
    def _fit_one_model(self, masks, margins, sample_index: int, seed_fit: int, seed_shuffle: int, n_train: int, n_test: int):
        """
        Fit one datamodel (i.e. for one sample).
        """
        X, y = self._get_samples_for_model(masks, margins, sample_index)
        X, y = shuffle(X, y, random_state=seed_shuffle)
        model = self.datamodel_factory.build_model(seed=seed_fit)
        X_train, y_train = self._train_samples(X, y, n_train)
        model.fit(X_train, y_train)
        X_test, y_test = self._test_samples(X, y, n_train, n_test)
        y_hat = model.predict(X_test)
        correlation = pearsonr(y_test, y_hat)[0]
        rmse = root_mean_squared_error(y_test, y_hat)
        
        return {'weights': model.coef_, 
                'bias': model.intercept_, 
                'correlation': correlation,
                'rmse': rmse}
    

    def fit_datamodels(self, indices, seed, n_train, n_test: int=None, in_memory: bool=True):
        """
        Fit datamodels for a given set of indices.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility. Defaults to None. Allows the pipeline to optionally interface with batch IDs if jobs are run in parallel.
        n_train : int
            Number of training mask-margin pairs to use for training the datamodels.
        n_test : int
            Number of test mask-margin pairs to use for testing the datamodels. Defaults to None, in which case all remaining mask-margin pairs are used for testing.
        indices : List[int]
            List of indices specifying the held-out samples for which the datamodels should be fit.
        in_memory : bool
            Whether to fit the datamodels in memory or to save them to disk.

        Returns
        -------
        Union[Dict[str, Any], str]
            - If `in_memory=True`: returns a dictionary with model outputs.
            - If `in_memory=False`: returns a string path to the output location on disk.
        """
        rngs_fit, rngs_shuffle = self._rng(seed)
        masks = self._masks
        margins = self._margins

        weights = self._create_array(in_memory, None if in_memory else os.path.join(self.path_to_outputs, f"batch_{seed}_weights.npy"),
            np.float32, (len(indices), masks.shape[1])
        )
        biases = self._create_array(in_memory, None if in_memory else os.path.join(self.path_to_outputs, f"batch_{seed}_biases.npy"),
            np.float32, (len(indices),)
        )
        correlations = self._create_array(in_memory, None if in_memory else os.path.join(self.path_to_outputs, f"batch_{seed}_correlations.npy"),
            np.float32, (len(indices),)
        )
        rmse = self._create_array(in_memory, None if in_memory else os.path.join(self.path_to_outputs, f"batch_{seed}_rmse.npy"),
            np.float32, (len(indices),)
        )
        for i, sample_index in tqdm(enumerate(indices), total=len(indices)):
            if not correlations[i]==0:
                continue
            model = self._fit_one_model(masks, margins, sample_index, rngs_fit.integers(0, 2**32 - 1), rngs_shuffle.integers(0, 2**32 - 1), n_train, n_test)
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
            np.save(os.path.join(self.path_to_outputs, f"batch_{seed}_sample_indices.npy"), indices, allow_pickle=False)
            return self.path_to_outputs

        
