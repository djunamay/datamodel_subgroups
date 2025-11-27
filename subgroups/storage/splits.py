
from dataclasses import dataclass, field
from typing import Dict
import numpy as np
import random
import string
from numpy.typing import NDArray

@dataclass
class Split:
    data: NDArray[np.int_]
    finished: bool = False 
    
@dataclass
class SplitStorage:
    splits: Dict[str, Split] = field(default_factory=dict)

    @property     
    def generate_barcode(self):
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choices(chars, k=8))
        
    def replace_split(self, split_name: str, new_split_1, new_split_2):
        self.splits[self.generate_barcode] = Split(data=new_split_1)
        self.splits[self.generate_barcode] = Split(data=new_split_2)
        del self.splits[split_name]

    def finish_split(self, split_name):
        self.splits[split_name].finished = True

    @property
    def _unfinished_splits(self):
        return [not self.splits[key].finished for key in self.splits.keys()]

    @property
    def _split_names(self):
        return np.array(list(self.splits.keys()))

    @property
    def unfinished_splits(self):
        return self._split_names[self._unfinished_splits]