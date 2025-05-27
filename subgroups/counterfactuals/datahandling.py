import numpy as np
class SplitClass:

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def X(self) -> np.ndarray:
        return self.X   

    def y(self) -> np.ndarray:
        return self.y

class CoarseSplits:

    def __init__(self, features: np.ndarray[float], labels: np.ndarray[bool], fine_label_bool: np.ndarray[bool]):
        self.features = features
        self.fine_label_bool = fine_label_bool
        self.labels = labels if len(np.unique(self.fine_label_bool[labels]))>1 else ~labels

    @property
    def whole(self) -> SplitClass:
        return SplitClass(X = self.features[~self.fine_label_bool & ~self.labels], y = self.labels[~self.fine_label_bool & ~self.labels])

    @property
    def split_a(self) -> SplitClass:
        return SplitClass(X = self.features[~self.fine_label_bool & self.labels], y = self.labels[~self.fine_label_bool & self.labels])

    @property
    def split_b(self) -> SplitClass:
        return SplitClass(X = self.features[self.fine_label_bool], y = self.labels[self.fine_label_bool])
