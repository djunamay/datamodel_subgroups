import glob
import numpy as np

def find_files_with_suffix(directory, suffix):
    pattern = f"{directory}/*{suffix}.npy"
    return sorted(glob.glob(pattern))


def load_eval_data(path, eval_name):
    files = np.array(find_files_with_suffix(path, eval_name))
    batch_number = np.array([int(ix.split('_')[-3]) for ix in files])
    order = np.argsort(batch_number)
    files_ordered = files[order]
    return np.hstack([np.load(ix) for ix in files_ordered]), batch_number[order]


def load_weights_data(path):
    files = np.array(find_files_with_suffix(path, 'weights'))
    batch_number = np.array([int(ix.split('_')[-2]) for ix in files])
    order = np.argsort(batch_number)
    files_ordered = files[order]
    return np.vstack([np.load(ix) for ix in files_ordered]), batch_number[order]