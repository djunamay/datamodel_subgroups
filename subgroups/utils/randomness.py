from numpy.random import Generator
import numpy as np

def generate_rngs_from_seed(seed: int, rng_names: list[str]) -> dict[str, Generator]:
    # one Generator per parameter so draws are independent yet reproducible
    seq = np.random.SeedSequence(seed)
    _rngs: dict[str, Generator] = {
        name: np.random.default_rng(s)   # sub-seeds
        for name, s in zip(
            rng_names,
            seq.spawn(len(rng_names))
        )
    }
    return _rngs