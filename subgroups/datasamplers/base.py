from typing import Protocol
from numpy.typing import NDArray
import numpy as np

class MaskGenerator(Protocol):
    """
    A function that generates a boolean mask indexing randomly selected samples for training, including an equal number of samples from each class.
    """
    def __call__(self, class_indices_0: NDArray[int], class_indices_1: NDArray[int], alpha: float, seed: int, num_samples: int) -> NDArray[bool]:
        ...

def generate_mask(class_indices_0: NDArray[int], class_indices_1: NDArray[int], alpha: float, seed: int, num_samples: int) -> NDArray[bool]:
    '''
    Returns a boolean mask indexing randomly selected samples for training, including an equal number of samples from each class.
    '''
    rng = np.random.default_rng(seed)
    mask = np.zeros(num_samples, dtype=bool)
    samples_per_class = int((num_samples*alpha)/2)
    indices_class_0, indices_class_1 = rng.permutation(class_indices_0)[:samples_per_class], rng.permutation(class_indices_1)[:samples_per_class]
    mask[np.concatenate([indices_class_0, indices_class_1])] = True
    return mask