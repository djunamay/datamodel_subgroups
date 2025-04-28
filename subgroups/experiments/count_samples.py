from subgroups.datasets.base import DatasetInterface
import chz 
from typing import Callable

#  python -m subgroups.experiments.count_samples dataset_factory=subgroups.datasets.registry:gtex

@chz.chz
class CountSamplesArgs:
    dataset_factory: Callable[[], DatasetInterface] = chz.field(
        doc="Callable that returns a dataset instance"
    )

def count_samples(args: CountSamplesArgs):
    ds = args.dataset_factory()  
    print(len(ds.features))
    
if __name__ == "__main__":
    chz.nested_entrypoint(count_samples)

# can have code for a pipeline that combines multiple takss, can import main fssrom count_samples
# because multiple tasks will use the same dataset, instead of setting it as an arg multiple times,
# chz lets you do x =...dataset 