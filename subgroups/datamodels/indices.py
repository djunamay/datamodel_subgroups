from typing import List
import chz

@chz.chz
class IndicesFunction:
    """
    A function that returns a list of sample indices to fit the datamodels for.
    Allows the user flexibility in deciding how to select indices to fit (e.g. may want to select only indices from one of the two coarse label classes).
    Batch_starter_seed allows interfacing with batch IDs if jobs are run in parallel.
    """
    def __call__(self, batch_starter_seed: int) -> List[int]:
        ...

@chz.chz
class SequentialIndices(IndicesFunction):
    """
    This implementation of IndicesFunction returns a list of indices to fit the datamodels for.
    The indices are selected sequentially from the total number of samples, with a batch size specified by the user.
    E.g. for batch size = 50, a call to the function with batch_starter_seed = 0 will return the first 50 indices, a call with batch_starter_seed = 1 will return the next 50 indices, and so on.
    """
    batch_size: int = chz.field()

    def __call__(self, batch_starter_seed: int) -> List[int]:
        start = batch_starter_seed * self.batch_size
        end = start + self.batch_size
        return list(range(start, end))


    