import json
import numpy as np
import glob
import pandas as pd

def load_model_architecture_at_index(file_path, index):
    """
    Loads a specific architecture dictionary from a newline-delimited JSON file.
    
    Parameters:
        file_path (str or Path): Path to the NDJSON file
        index (int): Index of the desired architecture (0-based)
    
    Returns:
        dict: The architecture dictionary at the specified index
    
    Raises:
        IndexError: If index is out of bounds
    """
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
        raise IndexError(f"Index {index} is out of bounds for the file.")


def load_json_numbers_as_array(file_path):
    """
    Loads a newline-delimited JSON file containing numbers (one per line)
    and returns them as a NumPy array.
    """
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return np.array(data)

def find_files_with_prefix(directory, prefix):
    pattern = f"{directory}/{prefix}*.json"
    return sorted(glob.glob(pattern))

def find_test_accuracy_files(directory):
    pattern = f"{directory}/test_accuracy_*.json"
    return sorted(glob.glob(pattern))

def find_snr_files(directory):
    pattern = f"{directory}/snr_*.json"
    return sorted(glob.glob(pattern))

def find_mask_factory_files(directory):
    pattern = f"{directory}/mask_factory_*.json"
    return sorted(glob.glob(pattern))

def return_mask_factory_data(directory):
    mask = load_json_numbers_as_array(directory)
    mask = [i['alpha'] for i in mask]
    index = int(directory.split('.json')[0].split('_')[-1])
    batch = np.repeat(index, len(mask))
    data = np.concatenate([batch.reshape(-1,1), np.array(mask).reshape(-1,1)], axis=1)
    return data

def return_test_accuracy_data(directory):
    accs = load_json_numbers_as_array(directory)
    index = int(directory.split('.json')[0].split('_')[-1])
    batch = np.repeat(index, len(accs))
    data = np.concatenate([batch.reshape(-1,1), accs.reshape(-1,1)], axis=1)
    return data

def return_snr_data(directory):
    snrs = load_json_numbers_as_array(directory)
    index = int(directory.split('.json')[0].split('_')[-1])
    batch = np.repeat(index, len(snrs))
    data = np.concatenate([batch.reshape(-1,1), snrs.reshape(-1,1)], axis=1)
    return data

def return_snr_alpha_data(directory):
    snr_paths = find_snr_files(directory)
    mask_factory_paths = find_mask_factory_files(directory)
    snr_data = np.concatenate([return_snr_data(path) for path in snr_paths])
    mask_factory_data = np.concatenate([return_mask_factory_data(path) for path in mask_factory_paths])
    data = np.concatenate([snr_data, mask_factory_data], axis=1)
    data = pd.DataFrame(data[:,[0,1,3]], columns=['batch', 'snr', 'alpha'])
    return data

def return_best_model_index(directory, acc_cutoff=0.8):
    # load all data
    acc_paths = find_test_accuracy_files(directory)
    snr_paths = find_snr_files(directory)
    test_accuracy_data = np.concatenate([return_test_accuracy_data(path) for path in acc_paths])
    snr_data = np.concatenate([return_snr_data(path) for path in snr_paths])

    # get indices for classifiers with test accuracy > cutoff
    index = np.argwhere(test_accuracy_data[:,1]>acc_cutoff).reshape(-1)

    # of these indices, get the one with the highest SNR
    snr_values = snr_data[index][:,1]
    best_snr = np.max(snr_values)

    if best_snr == 0:
        raise ValueError("Max SNR at this accuracy cutoff is 0. Don't proceed with this model.")
    best_model_index = index[np.argmax(snr_values)]

    # get the batch that this model is in
    best_model_batch = snr_data[:,0][best_model_index]
    best_model_index_in_batch = np.where(snr_data[snr_data[:,0]==best_model_batch][:,1]==best_snr)[0]
    return int(best_model_batch), int(best_model_index_in_batch)

def return_best_model_architecture(directory, acc_cutoff=0.8):
    best_model_batch, best_model_index_in_batch = return_best_model_index(directory, acc_cutoff)
    best_model_architecture = load_model_architecture_at_index(directory + f'/model_factory_{int(best_model_batch)}.json', best_model_index_in_batch)
    best_alpha = load_model_architecture_at_index(directory + f'/mask_factory_{int(best_model_batch)}.json', best_model_index_in_batch)
    return best_model_architecture, best_alpha

def return_best_model_test_accuracy(directory, best_model_batch, best_model_index_in_batch):
    best_test_accuracy = load_json_numbers_as_array(directory + f'/test_accuracy_{int(best_model_batch)}.json')[best_model_index_in_batch]
    return best_test_accuracy

def return_best_model_snr(directory, best_model_batch, best_model_index_in_batch):
    best_snr = load_json_numbers_as_array(directory + f'/snr_{int(best_model_batch)}.json')[best_model_index_in_batch]
    return best_snr

