from ..datasets.test_data import RandomDataset
from ..counterfactuals.datahandling import CoarseSplits
import numpy as np

def test_coarse_splits():
    data = RandomDataset()

    for i in np.unique(data.fine_labels):
        fine_labels = data.fine_labels==int(i)
        splits = CoarseSplits(features=data.features, labels=data.coarse_labels, fine_label_bool=fine_labels)

        # fine labels are a subset of coarse labels
        assert len(np.unique(data.coarse_labels[fine_labels]))==1

        # check the dimensions of the splits
        vals, counts = np.unique(fine_labels[data.coarse_labels], return_counts=True)
        vals2, counts2 = np.unique(fine_labels[np.invert(data.coarse_labels)], return_counts=True)

        if len(counts2)==1:
            assert splits.whole.X.shape[0] == counts2[0]
            assert splits.split_a.X.shape[0] == counts[~vals]
            assert splits.split_b.X.shape[0] == counts[vals]
        elif len(counts2)==2:
            assert splits.whole.X.shape[0] == counts[0]
            assert splits.split_a.X.shape[0] == counts2[~vals2]
            assert splits.split_b.X.shape[0] == counts2[vals2]
