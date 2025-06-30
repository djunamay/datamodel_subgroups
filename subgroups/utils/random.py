import numpy as np

def fork_rng(rng: np.random.Generator, n_children: int) -> list[np.random.Generator]:
    return [np.random.default_rng(child) for child in rng.bit_generator.seed_seq.spawn(n_children)]