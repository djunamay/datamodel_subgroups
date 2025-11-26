from collections.abc import Mapping, Sequence
import json
import chz
import os
from pathlib import Path
import numpy as np
import functools

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


def write_chz_class_to_json(obj, file_path: str | os.PathLike, *, indent: int = 2, append: bool = True):
    """
    Serialize any CHZ object to JSON.
    Non-JSON types are converted like this:
        • numpy scalar  -> native Python int/float
        • numpy ndarray -> nested list (slow, use with care)
        • classes / functions -> "module:qualname" string
        • everything else -> str(o)
    """
    def _fallback(o):
        # 1️numpy scalars
        if isinstance(o, (np.integer, np.floating)):
            return o.item()

        # 2️ small 1-D/2-D ndarrays (optional – remove if huge)
        if isinstance(o, np.ndarray):
            return o.tolist()

        if isinstance(o, functools.partial):
            # minimal encoding: just record the underlying function
            return {
                "__type__": "partial",
                "func_module": o.func.__module__,
                "func_qualname": o.func.__qualname__,
                "args": o.args,
                "keywords": o.keywords or {},
            }


        # 3️ class objects or callables
        if isinstance(o, type) or callable(o):
            return f"{o.__module__}:{o.__qualname__}"

        if hasattr(o, "__class__") and o.__class__.__module__ != "builtins":
            return f"{o.__class__.__module__}.{o.__class__.__qualname__}"
        # 4️ any CHZ object -> its pretty repr without colours
        if hasattr(o, "__chz_pretty__"):
            return o.__chz_pretty__(colored=False)

        # 5️ final fall-through
        return str(o)

    data = chz.beta_to_blueprint_values(obj)          # regular chz → dict conversion
    if os.path.exists(file_path) and append:
        mode = "a"
    else:
        mode = "w"
    with open(file_path, mode) as fp:
        json.dump(data, fp, default=_fallback, indent=indent)
        fp.write("\n")


def deep_equal(a, b):
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, Sequence) and not isinstance(a, (str, bytes)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b

def check_and_write_config(experiment, path_to_config, overwrite: bool=False, append: bool=False):
    """
    Check if the experiment config has changed from previous run.
    """
    if overwrite:
        write_chz_class_to_json(experiment, path_to_config, append=append)

    elif os.path.exists(path_to_config):
            previous_experiment_config = load_experiment_from_json(path_to_config)
            path_to_tmp_config = os.path.join(experiment.path_to_results, 'tmp.json')
            write_chz_class_to_json(experiment, path_to_tmp_config)
            current_experiment_config = load_experiment_from_json(path_to_tmp_config)
            os.remove(path_to_tmp_config)
            if not deep_equal(previous_experiment_config, current_experiment_config):
                raise RuntimeError("Experiment config has changed from previous run. If this is intentional, set overwrite_config=True")
    else:
        write_chz_class_to_json(experiment, path_to_config, append=append)

def append_chz_ndjson(chz_class, file_path: str):
    """Append one CHZ object as a line of NDJSON."""
    with open(file_path, "a") as f:               # 'a' = append
        json.dump(chz.asdict(chz_class), f)
        f.write("\n")     

def append_float_ndjson(value: float, file_path: Path) -> None:
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