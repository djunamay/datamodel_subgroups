import glob
import numpy as np
import chz

def find_files_with_suffix(directory, suffix):
    pattern = f"{directory}/*{suffix}"
    return sorted(glob.glob(pattern))


def print_n_models(path, suffix='test_accuracies.npy'):
    files = find_files_with_suffix(path, suffix)
    n_models = np.sum([np.sum(np.load(file)!=0) for file in files])
    print(n_models, ' models were trained.')

if __name__ == "__main__":
    chz.entrypoint(print_n_models)