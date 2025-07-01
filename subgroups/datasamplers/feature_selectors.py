import chz
import numpy as np
from numpy.typing import NDArray
from .base import SelectPCsInterface

@chz.chz
class SelectPCsBasic(SelectPCsInterface):
    def feature_indices(self, n_pcs: int) -> NDArray[int]:
        return np.arange(n_pcs)
    
@chz.chz
class SelectPCsSingleCell(SelectPCsInterface):
    nfeatures: int = chz.field(default=1600, doc="Number of features")
    ncelltypes: int = chz.field(default=8, doc="Number of cell types")
    ndim: int = chz.field(default=50, doc="Number of dimensions")
    nstats: int = chz.field(default=4, doc="Number of statistics")

    def feature_indices(self, n_pcs: int) -> NDArray[int]:
        pcs_sele = n_pcs*self.nstats
        pcs_sele = np.arange(pcs_sele)
        cols_per_celltype = self.ndim*self.nstats
        return np.hstack([pcs_sele + (cols_per_celltype*celltype) for celltype in range(self.ncelltypes)])