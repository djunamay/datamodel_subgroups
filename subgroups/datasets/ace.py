import numpy as np
import pandas as pd
from .base import BaseDataset  
import chz  
from numpy.typing import NDArray

@chz.chz
class AceDataset(BaseDataset):
    """ACE dataset: process this specific dataset."""
    path_to_data: str=chz.field(doc="Path to the CSF data")
    path_to_meta_data: str=chz.field(doc="Path to the meta data")
    
    @chz.init_property
    def _full_csf_data(self):
        csf_data = pd.read_csv(self.path_to_data, sep='\t', low_memory=False)
        code = csf_data['csf_code']
        csf_data = csf_data[[col for col in csf_data.columns if col.endswith('CSF')]]
        csf_data.index = code
        return csf_data.dropna(axis=0)

    @chz.init_property
    def _full_meta_data(self):
        meta_data = pd.read_csv(self.path_to_meta_data, sep='\t', usecols=range(1971), low_memory=False)
        meta_data.index = meta_data['csf_code']
        # syndromic tag spanish -> english
        dictionary = dict(zip(meta_data['diagnostic_syndromic_csf_tag'].unique(), ['mild_cognitive_impairment', 'dementia', 'subjective_memory_complaint', 'control', 'other', 'other']))
        meta_data['syndromic_tag'] = meta_data['diagnostic_syndromic_csf_tag'].map(dictionary)
        # convert dates to datetime
        for col in ['date_of_birth', 'date_csf', 'date_monitoring_csf']:
            meta_data[col] = pd.to_datetime(meta_data[col], format='%Y-%m-%d')
        # calculate age and time difference
        meta_data['age'] = meta_data.apply(lambda row: self._calculate_age(row['date_of_birth'], row['date_csf']), axis=1)
        meta_data['abs_time_cog_to_csf_days'] = np.abs((meta_data['date_csf']-meta_data['date_monitoring_csf']).dt.days)
        return meta_data 

    @chz.init_property
    def _indices_to_keep(self):
        meta_data = self._full_meta_data.loc[self._full_csf_data.index]
        indices_to_keep = (
            (meta_data["abs_time_cog_to_csf_days"] < 155)
            & meta_data["syndromic_tag"].isin({"mild_cognitive_impairment", "dementia"})
        )
        return indices_to_keep

    @staticmethod
    def _calculate_age(birth_date, reference_date):
        age = reference_date.year - birth_date.year
        if (reference_date.month, reference_date.day) < (birth_date.month, birth_date.day):
            age -= 1
        return age

    @property
    def _descriptive_data(self):
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

    @property
    def _csf_data(self):
        return self._full_csf_data.loc[self._indices_to_keep].values.astype(float)

    @property
    def _meta_data(self):
        return self._full_meta_data.loc[self._indices_to_keep] 
    
    @property
    def features(self)-> NDArray[float]:
        return self._csf_data

    @property
    def coarse_labels(self)-> NDArray[bool]:
        return (self._meta_data['syndromic_tag'] == 'dementia').values.astype(bool)
    
    @property
    def fine_labels(self)-> NDArray[bool]:
        x = self._meta_data['diagnostic_primary_csf_label'].unique()
        dictionary = dict(zip(x, range(len(x))))
        return self._meta_data['diagnostic_primary_csf_label'].map(dictionary).values.astype(int)
    
    @property
    def descriptive_data(self)-> np.recarray:
        return self._descriptive_data
    
    
