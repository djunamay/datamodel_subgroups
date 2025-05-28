import numpy as np
import pandas as pd
from .base import BaseDataset  
import chz  
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

@chz.chz
class AceDataset(BaseDataset):
    """
    Processes the ACE dataset, providing access to features, labels, and metadata.

    Attributes
    ----------
    path_to_data : str
        Path to the CSF data.
    path_to_meta_data : str
        Path to the meta data.
    """
    path_to_data: str = chz.field(doc="Path to the CSF data")
    path_to_meta_data: str = chz.field(doc="Path to the meta data")
    
    @chz.init_property
    def _full_csf_data(self):
        """
        Dataframe with preprocessed CSF data with 'csf_code' as index.
        """
        csf_data = pd.read_csv(self.path_to_data, sep='\t', low_memory=False)
        code = csf_data['csf_code']
        csf_data = csf_data[[col for col in csf_data.columns if col.endswith('CSF')]]
        csf_data.index = code
        return csf_data.dropna(axis=0)

    @chz.init_property
    def _full_meta_data(self):
        """
        Dataframe with preprocessed meta data, including date conversion and age calculation.
        """
        meta_data = pd.read_csv(self.path_to_meta_data, sep='\t', usecols=range(1971), low_memory=False)
        meta_data.index = meta_data['csf_code']
        dictionary = dict(zip(meta_data['diagnostic_syndromic_csf_tag'].unique(), ['mild_cognitive_impairment', 'dementia', 'subjective_memory_complaint', 'control', 'other', 'other']))
        meta_data['syndromic_tag'] = meta_data['diagnostic_syndromic_csf_tag'].map(dictionary)
        for col in ['date_of_birth', 'date_csf', 'date_monitoring_csf']:
            meta_data[col] = pd.to_datetime(meta_data[col], format='%Y-%m-%d')
        meta_data['age'] = meta_data.apply(lambda row: self._calculate_age(row['date_of_birth'], row['date_csf']), axis=1)
        meta_data['abs_time_cog_to_csf_days'] = np.abs((meta_data['date_csf']-meta_data['date_monitoring_csf']).dt.days)
        return meta_data 

    @chz.init_property
    def _indices_to_keep(self):
        """
        Boolean series indicating which samples to keep.
        """
        meta_data = self._full_meta_data.loc[self._full_csf_data.index]
        indices_to_keep = (
            (meta_data["abs_time_cog_to_csf_days"] < 155)
            & meta_data["syndromic_tag"].isin({"mild_cognitive_impairment", "dementia"})
        )
        return indices_to_keep

    @staticmethod
    def _calculate_age(birth_date, reference_date):
        """
        Calculate age based on birth date and a reference date.

        Parameters
        ----------
        birth_date : datetime
            Date of birth.
        reference_date : datetime
            Reference date for age calculation.

        Returns
        -------
        int
            Calculated age.
        """
        age = reference_date.year - birth_date.year
        if (reference_date.month, reference_date.day) < (birth_date.month, birth_date.day):
            age -= 1
        return age
    
    @staticmethod
    def _reduce_dimensionality(array: NDArray[float], n_components: int = 50) -> NDArray[float]:
        """
        Reduce the dimensionality of the given array using PCA.

        Parameters
        ----------
        array : NDArray[float]
            The data array to be reduced.
        n_components : int, optional
            Number of components for PCA. Default is 50.

        Returns
        -------
        NDArray[float]
            The array with reduced dimensionality.
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(array)
        pca = PCA(n_components=n_components)
        projections = pca.fit_transform(scaled_data)
        return projections
    
    @property
    def _descriptive_data(self):
        """
        Record array of descriptive metadata fields.
        """
        vars = [
            'age',
            'abs_time_cog_to_csf_days', 
            'csf_technique',
            'csf_abeta_42',
            'csf_p_tau',
            'csf_tau',
            'csf_ratio_abeta_42_40', 
            'a_1pos_0neg',
            't_1pos_0neg',
            'n_1pos_0neg',
            'sex_1M_2F',
            'genetic_apoe',
            'mmse_csf',
            'cdr_csf',
            'gds_csf',
            'syndromic_tag', 
            'diagnostic_primary_csf_label'
        ]
        return self._meta_data[vars].to_records(index=False)

    @chz.init_property
    def _csf_data(self):
        """
        Filtered CSF data.
        """
        return self._reduce_dimensionality(self._full_csf_data.loc[self._indices_to_keep].values.astype(float))

    @property
    def _meta_data(self):
        """
        Filtered meta data.
        """
        return self._full_meta_data.loc[self._indices_to_keep] 
    
    @property
    def features(self) -> NDArray[float]:
        """
        Feature matrix (shape: [n_samples, n_features]).
        """
        return self._csf_data

    @property
    def coarse_labels(self) -> NDArray[bool]:
        """
        Binary labels indicating dementia status (shape: [n_samples]).
        """
        #return (self._meta_data['syndromic_tag'] == 'dementia').values.astype(bool)
        coarse_labels = [x in set([130100.0, 130200.0, 130400.0]) for x in self._meta_data['diagnostic_primary_csf']] # amnestic vs non-amnestic split recommended by Diane Chan
        return np.array(coarse_labels, dtype=bool)
    
    @property
    def fine_labels(self) -> NDArray[bool]:
        """
        Integer labels for each unique primary CSF diagnostic label (shape: [n_samples]).
        """
        x = self._meta_data['diagnostic_primary_csf_label'].unique()
        dictionary = dict(zip(x, range(len(x))))
        return self._meta_data['diagnostic_primary_csf_label'].map(dictionary).values.astype(int)
    
    @property
    def descriptive_data(self) -> np.recarray:
        """
        Descriptive data (shape: [n_samples, n_descriptive_features]).
        """
        return self._descriptive_data
