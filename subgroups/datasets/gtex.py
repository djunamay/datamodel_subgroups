import numpy as np
import pandas as pd
from .base import BaseDataset  
import chz  
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

@chz.chz
class GTEXDataset(BaseDataset):
    """GTEX dataset: process this specific dataset."""
    path_to_data: str
    path_to_meta_data: str
    path_to_sample_metadata: str
    n_components: int=chz.field(doc="Number of components to reduce the dimensionality to", default=50)
    
    @chz.init_property
    def _tpm_data(self):
        tpm_data = pd.read_csv(self.path_to_data, sep='\t', skiprows=2)
        tmp_data = tpm_data.iloc[:,2:].T
        index = tmp_data.index
        tmp_data = self._reduce_dimensionality(tmp_data.values.astype(float), self.n_components)
        return tmp_data, index
    
    @chz.init_property
    def _meta_data(self):
        meta_data = pd.read_csv(self.path_to_meta_data, sep='\t', low_memory=False)
        meta_data.index = meta_data['SAMPID']
        meta_data = meta_data.loc[self._tpm_data[1]]
        subject_metadata = pd.read_csv(self.path_to_sample_metadata, sep='\t')
        dictionaries = [dict(zip(subject_metadata['SUBJID'], subject_metadata[x])) for x in subject_metadata.columns[1:]]
        meta_data['SUBJID'] = ['-'.join(x.split('-',2)[:2]) for x in meta_data['SAMPID']]
        for i in range(len(dictionaries)):
            meta_data[subject_metadata.columns[1:][i]] = meta_data['SUBJID'].map(dictionaries[i])
        return meta_data

    @staticmethod
    def _reduce_dimensionality(array: NDArray[float], n_components: int=50)-> NDArray[float]:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(array)
        pca = PCA(n_components=n_components)
        projections = pca.fit_transform(scaled_data)
        return projections

    @property
    def _descriptive_data(self):
        vars = ['SAMPID',
                'SMNABTCH',
                'SMRIN', 
                'SUBJID',
                'SEX',
                'AGE',
                'DTHHRDY']
        return self._meta_data[vars].to_records(index=False)
    
    @property
    def features(self)-> NDArray[float]:
        return self._tpm_data[0]

    @property
    def coarse_labels(self)-> NDArray[bool]: 
        return (self._meta_data['SMTS'] == 'Brain').values.astype(bool)
    
    @property
    def fine_labels(self)-> NDArray[bool]:
        x = self._meta_data['SMTSD'].unique()
        dictionary = dict(zip(x, range(len(x))))
        return self._meta_data['SMTSD'].map(dictionary).values.astype(int)

    @property
    def descriptive_data(self)-> np.recarray:
        return self._descriptive_data