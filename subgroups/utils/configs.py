from collections.abc import Mapping, Sequence
import json
import chz
import os
def load_experiment_from_json(file_path: str) -> dict:
    """
    Load experiment configuration from a JSON file.

    Parameters
    ----------
    file_path : str
        Path to the JSON file containing the experiment configuration.

    Returns
    -------
    dict
        Dictionary containing the experiment configuration.
    """
    with open(file_path, 'r') as f:
        experiment_config = json.load(f)
    return experiment_config

def write_chz_class_to_json(chz_class, file_path: str):
    """
    Write a CHZ class to a JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(chz.asdict(chz_class), f)

def deep_equal(a, b):
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, Sequence) and not isinstance(a, (str, bytes)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b

def check_and_write_config(experiment, path_to_config, overwrite: bool=False):
    """
    Check if the experiment config has changed from previous run.
    """
    if overwrite:
        write_chz_class_to_json(experiment, path_to_config)
    elif os.path.exists(path_to_config):
            experiment_config = load_experiment_from_json(path_to_config)
            if not deep_equal(experiment_config, chz.asdict(experiment)):
                raise RuntimeError("Experiment config has changed from previous run.\n If this is intentional, set overwrite=True")
    else:
        write_chz_class_to_json(experiment, path_to_config)