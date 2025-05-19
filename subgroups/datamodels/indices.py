from typing import List

class IndicesFunction:
    def __call__(self, batch_starter_seed: int) -> List[int]:
        ...


class SequentialIndices(IndicesFunction):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def __call__(self, batch_starter_seed: int) -> List[int]:
        start = batch_starter_seed * self.batch_size
        end = start + self.batch_size
        return list(range(start, end))


    