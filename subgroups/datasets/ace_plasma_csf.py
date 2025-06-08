import numpy as np
import pandas as pd
from .base import BaseDataset  
import chz  
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

@chz.chz
class AceDatasetPlasmaCSF(BaseDataset):
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
    path_to_sample_meta_data: str = chz.field(doc="Path to the meta data")
    path_to_feature_meta_data_csf: str = chz.field(doc="Path to the feature info", default='/home/Genomica/03-Collabs/djuna/data/HARPONE-Somalogic_CSF_Annotations_anmlSMP.xlsx')
    path_to_feature_meta_data_plasma: str = chz.field(doc="Path to the feature info", default='/home/Genomica/03-Collabs/djuna/data/HARPONE-Somalogic_Plasma_Annotations_anmlSMP.xlsx')
    path_to_sample_meta_data_dictionary: str = chz.field(doc="Path to the data dictionary", default='/home/Genomica/03-Collabs/djuna/clinical_data_ACE/202406_FACE_data_dictionary_CSF.xlsx')
    split: str = chz.field(doc="coarse label to use for training", default='amnestic')
    
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
    
    @staticmethod
    def _feature_info(path_to_feature_meta_data):
        """
        Load feature info from excel file.
        """
        feature_info = pd.read_excel(path_to_feature_meta_data)
        feature_info.index = feature_info['AptName']
        return feature_info
    
    @staticmethod
    def _filter_features(feature_info, suffix):
        """
        Keep only protein features that are human and have passed the column check.
        """
        features_keep = (feature_info['Type']=='Protein') & (feature_info['Organism']=='Human') & (feature_info['ColCheck']=='PASS') & (feature_info['EntrezGeneSymbol']!='APP') & (feature_info['EntrezGeneSymbol']!='MAPT') # removing these features as derivatives used to annoate amnestic labels
        features_keep =  feature_info['AptName'][np.array(features_keep)]
        return [x+'.SOMA_HARP2021_'+suffix for x in features_keep]

    @chz.init_property
    def _csf_feature_info(self):
        """
        Load feature info from excel file.
        """
        return self._feature_info(self.path_to_feature_meta_data_csf)
    
    @chz.init_property
    def _plasma_feature_info(self):
        """
        Load feature info from excel file.
        """
        return self._feature_info(self.path_to_feature_meta_data_plasma)
    
    @chz.init_property
    def _csf_features_to_keep(self):
        """
        Keep only protein features that are human and have passed the column check.
        """
        return self._filter_features(self._csf_feature_info, 'CSF')
    
    @chz.init_property
    def _plasma_features_to_keep(self):
        """
        Keep only protein features that are human and have passed the column check.
        """
        return self._filter_features(self._plasma_feature_info, 'PLASMA')
    
    @chz.init_property
    def _csf_columns_to_keep(self):
        """
        Keep only the columns that are in the CSF data.
        """
        print(self._plasma_features_to_keep)
        print(self._csf_features_to_keep)
        return np.intersect1d(self._plasma_features_to_keep, self._csf_features_to_keep)
    
    @chz.init_property
    def _full_csf_data(self):
        """
        Dataframe with preprocessed CSF data with 'csf_code' as index.
        """
        csf_data = pd.read_csv(self.path_to_data, sep='\t', low_memory=False)
        csf_data = csf_data.dropna(axis=0)
        code = csf_data['csf_code']
        csf_data = csf_data[self._csf_columns_to_keep]
        csf_data.index = code
        return csf_data

    @chz.init_property
    def _meta_data_description(self):
        """
        Load data dictionary that describes the clinical variables in the meta data from excel file.
        """
        data_dictionary = pd.read_excel(self.path_to_sample_meta_data_dictionary)
        data_dictionary.index = data_dictionary['variable_name']
        return data_dictionary
    
    @chz.init_property
    def _full_meta_data(self):
        """
        Load meta data from csv file.
        """
        meta_data = pd.read_csv(self.path_to_sample_meta_data, sep='\t', usecols=range(1971), low_memory=False)
        meta_data.index = meta_data['csf_code']
        return meta_data
    
    @chz.init_property
    def _meta_data_columns_to_keep(self):
        """
        Keep only the meta data columns that have descriptions.
        """
        return np.intersect1d(self._full_meta_data.columns, self._meta_data_description['variable_name']) # only keep metadata variables that have descriptions
    
    @chz.init_property
    def _meta_data(self):
        """
        Dataframe with preprocessed meta data, including date conversion and age calculation.
        """
        meta_data = (self._full_meta_data[self._meta_data_columns_to_keep]).copy()
        dictionary = dict(zip(meta_data['diagnostic_syndromic_csf_tag'].unique(), ['mild_cognitive_impairment', 'dementia', 'subjective_memory_complaint', 'control', 'other', 'other']))
        meta_data['syndromic_tag'] = meta_data['diagnostic_syndromic_csf_tag'].map(dictionary)
        for col in ['date_of_birth', 'date_csf', 'date_monitoring_csf']:
            meta_data[col] = pd.to_datetime(meta_data[col], format='%Y-%m-%d')
        meta_data['age'] = meta_data.apply(lambda row: self._calculate_age(row['date_of_birth'], row['date_csf']), axis=1)
        meta_data['abs_time_cog_to_csf_days'] = np.abs((meta_data['date_csf']-meta_data['date_monitoring_csf']).dt.days)
        meta_data['age_group'] = meta_data['age']>np.median(meta_data['age'])
        return meta_data.loc[self._full_csf_data.index] 

    @chz.init_property
    def _samples_to_keep(self):
        """
        Boolean series indicating which samples to keep based on the time difference between the date of the CSF sample and the date of the cognitive assessment.
        """
        indices_to_keep = (
            (self._meta_data["abs_time_cog_to_csf_days"] < 155)
            & self._meta_data["syndromic_tag"].isin({"mild_cognitive_impairment", "dementia"})
        )
        return indices_to_keep

    @chz.init_property
    def _filtered_reduced_csf_data(self):
        """
        Reduced dimensionality CSF data and subsetted to the samples that are within 155 days of the cognitive assessment.
        """
        return self._reduce_dimensionality(self._full_csf_data.loc[self._samples_to_keep].values.astype(float))

    @property
    def _filtered_meta_data(self):
        """
        Subset the meta data to the samples that are within 155 days of the cognitive assessment.
        """
        return self._meta_data.loc[self._samples_to_keep] 
    
    @property
    def descriptive_data(self):
        """
        Record array of descriptive metadata fields.
        """
        return self._filtered_meta_data.to_records(index=False)

    @property
    def features(self) -> NDArray[float]:
        """
        Feature matrix (shape: [n_samples, n_features]).
        """
        return self._filtered_reduced_csf_data

    @property
    def coarse_labels(self) -> NDArray[bool]:
        """
        Binary labels indicating dementia or ae status (shape: [n_samples]).
        """
        if self.split == 'amnestic':
            coarse_labels = [x in set([130100.0, 130200.0, 130400.0]) for x in self._filtered_meta_data['diagnostic_primary_csf']] # amnestic vs non-amnestic split recommended by Diane Chan
        elif self.split == 'age_group':
            coarse_labels = self._filtered_meta_data['age_group']
        else:
            raise ValueError(f"Invalid split: {self.split}")
        return np.array(coarse_labels, dtype=bool)
    
    @property
    def fine_labels(self) -> NDArray[bool]:
        """
        Integer labels for each unique primary CSF diagnostic label (shape: [n_samples]).
        """
        
        x = self._filtered_meta_data['diagnostic_primary_csf_label'].unique()
        dictionary = dict(zip(x, range(len(x))))
        return self._filtered_meta_data['diagnostic_primary_csf_label'].map(dictionary).values.astype(int)

    @chz.init_property
    def data_dictionary(self):
        """
        Data dictionary for the sample meta data that is kept.
        """
        dictionary = self._meta_data_description.loc[self._meta_data_columns_to_keep]
        lookup = (
            dictionary.set_index("variable_name")      
            .to_dict(orient="index")         
        )
        return lookup
    