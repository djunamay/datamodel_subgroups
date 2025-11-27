from ..utils.pick_best_architecture import find_files_with_prefix, load_json_numbers_as_array
import numpy as np
import json
import chz
from pathlib import Path

def all_lists_equal(d: dict) -> bool:
    """
    Return True if every value in d is equal to the first value.
    """
    it = iter(d.values())
    try:
        first = next(it)
    except StopIteration:
        # empty dict → vacuously True
        return True
    return all(val == first for val in it)

def dict_to_recarray(data: dict) -> np.recarray:
    """
    Convert a dict of equal-length sequences into a NumPy recarray.

    Parameters
    ----------
    data : dict
        Keys are field names, values are sequences (lists, arrays, etc.)
        Must all be the same length.

    Returns
    -------
    recarr : np.recarray
        A record array where you can access columns as attributes.
    """
    names = list(data.keys())
    arrays = [np.asarray(data[name]) for name in names]

    # np.rec.fromarrays takes a list of arrays and a comma-sep string of names
    recarr = np.rec.fromarrays(arrays, names=','.join(names))
    return recarr

def combine_snr_outputs(path: str, output_path: str):
    file_prefix = ['n_pcs_', 'test_accuracy_', 'snr_', 'mask_factory_', 'model_factory_']

    data_numbers = {}
    data_batch = {}

    for file_prefix in file_prefix:
        files = find_files_with_prefix(path, file_prefix)
        numbers = [load_json_numbers_as_array(file) for file in files]
        numbers = np.concatenate(numbers)
        batch = [int(file.split('_')[-1].split('.')[0]) for file in files]
        if file_prefix == 'mask_factory_':
            numbers = [number['alpha'] for number in numbers]
            file_prefix='alpha_'
        if file_prefix == 'model_factory_':
            numbers = [number['max_depth'] for number in numbers]
            file_prefix='max_depth_'
        data_numbers[file_prefix] = numbers
        data_batch[file_prefix] = batch

    if all_lists_equal(data_batch):
        np.save(output_path, dict_to_recarray(data_numbers))
    else:
        raise ValueError("The number of batches and/or their order are not the same.")


if __name__ == "__main__":
    chz.entrypoint(combine_snr_outputs)
