from .base import DatamodelsPipelineInterface
from .regressor import DataModelFactory
from ..datastorage.base import CombinedMaskMarginStorageInterface
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
    combined_mask_margin_storage: CombinedMaskMarginStorageInterface = chz.field(doc='class containing the masks and margins.')
    datamodel_factory: DataModelFactory = chz.field(doc='Factory for creating the datamodel.')
    path_to_outputs: str = chz.field(default=None, doc='Path to save the datamodel outputs.')

    @property
    def _masks(self):
        return self.combined_mask_margin_storage.masks
    
    @property
    def _margins(self):
        return self.combined_mask_margin_storage.margins
    
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
        mask = masks[:,sample_index]
        index = ~mask
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
        if n_test is not None:
            masks = self._masks[:n_train+n_test]
            margins = self._margins[:n_train+n_test]
        else:
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
            fit_rng = rngs_fit.integers(0, 2**32 - 1)
            shuffle_rng = rngs_shuffle.integers(0, 2**32 - 1)
            if not correlations[i]==0:
                continue
            model = self._fit_one_model(masks, margins, sample_index, fit_rng, shuffle_rng, n_train, n_test)
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

        
