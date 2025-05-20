from typing import List
import chz

@chz.chz
class IndicesFunction:
    def __call__(self, batch_starter_seed: int) -> List[int]:
        ...

@chz.chz
class SequentialIndices(IndicesFunction):
    batch_size: int = chz.field()

    def __call__(self, batch_starter_seed: int) -> List[int]:
        start = batch_starter_seed * self.batch_size
        end = start + self.batch_size
        return list(range(start, end))


    