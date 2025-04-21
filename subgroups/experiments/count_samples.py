from subgroups.datasets import DatasetInterface, GTEXDataset, AceDataset
import chz 

# python -m subgroups.experiments.count_samples dataset=subgroups.datasets.registry.gtex

@chz.chz
class CountSamplesArgs:
    dataset: DatasetInterface=chz.field(doc="Dataset to count samples from")

def count_samples(args: CountSamplesArgs):
    print(len(args.dataset.features))
    
if __name__ == "__main__":
    chz.nested_entrypoint(count_samples)

# can have code for a pipeline that combines multiple takss, can import main fssrom count_samples
# because multiple tasks will use the same dataset, instead of setting it as an arg multiple times,
# chz lets you do x =...dataset 