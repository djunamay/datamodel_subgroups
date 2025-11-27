import chz
from functools import cached_property
import os, glob, hashlib
from abc import ABC
from typing import Union
import numpy as np
Array = Union[np.ndarray, np.memmap]
from .training import BaseStorage

@chz.chz
class CombinedMaskMarginStorage(BaseStorage):
    """
    A basic implementation of DatamodelsPipelineInterface, which fits a SklearnRegressor to each sample specified in the indices.
    The datamodels are fitted independently of one another.
    """
    path_to_inputs: str = chz.field(doc='Path to models outputs containing the masks and margins.')

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

    @staticmethod
    def _convert_masks_to_signatures(masks):
        signatures = []
        for m in masks:
            packed = np.packbits(m)        # uint8 array, each bit is one bool
            sig = hashlib.md5(packed).hexdigest()
            signatures.append(sig)
        return signatures
    
    @staticmethod
    def _completed_models_bool(srcs):
        temp =[src!=0 for src in srcs]
        return np.hstack(temp)
    
    @staticmethod
    def _combined_masks(completed_masks, duplicates_masks):
        mask = duplicates_masks.copy()
        mask[~completed_masks] = False
        return mask
    
    @staticmethod
    def _split_masks(mask, srcs):
        rows = [m.shape[0] for m in srcs]
        cut_points = np.cumsum(rows)[:-1]
        mask_chunks = np.split(mask, cut_points)
        assert mask.size == sum(rows), "Mask length mismatch"
        return mask_chunks

    def _mask_to_signatures(self, srcs):
        signatures_per_masks = np.hstack([self._convert_masks_to_signatures(src) for src in srcs])
        _, first_index = np.unique(signatures_per_masks, return_index=True)
        return np.isin(np.arange(len(signatures_per_masks)), first_index)
    
    @cached_property
    def _model_completed_masks(self):
        in_paths_acc = self._find_files(self.path_to_inputs, 'test_accuracies')
        srcs_acc  = [np.lib.format.open_memmap(p, mode="r") for p in in_paths_acc]
        completed_masks = self._completed_models_bool(srcs_acc)
        in_paths = self._find_files(self.path_to_inputs, 'masks')
        srcs  = [np.lib.format.open_memmap(p, mode="r") for p in in_paths]
        duplicates_masks = self._mask_to_signatures(srcs)
        mask = self._combined_masks(completed_masks, duplicates_masks)
        return self._split_masks(mask, srcs)
        
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
        model_mask_chunks = self._model_completed_masks
        total_chunks = [x.sum() for x in model_mask_chunks]
        orig_dtype = srcs[0].dtype
        out_dtype = np.float16 if orig_dtype == np.float32 else orig_dtype
        out = np.lib.format.open_memmap(out_path,
                                        mode="w+",
                                        dtype=out_dtype,
                                        shape=(int(np.sum(total_chunks)), srcs[0].shape[1]))   
        offset = 0
        for i, arr in enumerate(srcs):
            n = total_chunks[i]
            out[offset:offset+n] = arr[model_mask_chunks[i]].copy()    
            offset += n
        out._mmap.close() 

        for arr in srcs:
            arr._mmap.close()
        
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
    def masks(self):
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
    def margins(self):
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
    