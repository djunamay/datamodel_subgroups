from subgroups.datasets import DatasetInterface, GTEXDataset, AceDataset
import chz 
from count_samples import main as count_samples, CountSamplesArgs # we can run this stage without using bash, ideally never use bash
from train_model import main as train_model, TrainModelArgs # don't call it main

@chz.chz
class PipelineArgs:
    dataset: DatasetInterface=chz.field(doc="Dataset to count samples from")

def main(args: PipelineArgs):
    count_samples_instance = CountSamplesArgs(dataset=args.dataset)
    train_model_instance = TrainModelArgs(dataset=args.dataset) # so that dataset is specified only once
    count_samples(count_samples_instance)
    
if __name__ == "__main__":
    chz.nested_entrypoint(main)

# can have code for a pipeline that combines multiple takss, can import main from count_samples
# because multiple tasks will use the same dataset, instead of setting it as an arg multiple times,
# chz lets you do x =...dataset 