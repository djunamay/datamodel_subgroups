from collections.abc import Mapping, Sequence
import json
import chz
import os
from pathlib import Path

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

def append_chz_ndjson(chz_class, file_path: str):
    """Append one CHZ object as a line of NDJSON."""
    with open(file_path, "a") as f:               # 'a' = append
        json.dump(chz.asdict(chz_class), f)
        f.write("\n")     

def append_float_ndjson(value: float, file_path: str) -> None:
    """
    Append a float to an NDJSON file.  Every call writes **one** JSON value
    followed by a newline, so the file can be streamed line-by-line later.

    Example line in the file:
        3.141592653589793

    Parameters
    ----------
    value      : The float you want to store.
    file_path  : Target file (created if missing, appended otherwise).
    """
    file_path = Path(file_path)
    with file_path.open("a", encoding="utf-8") as f:   # 'a' → append
        json.dump(float(value), f)   # ensures 1.0 stays 1.0, not "1.0"
        f.write("\n")   