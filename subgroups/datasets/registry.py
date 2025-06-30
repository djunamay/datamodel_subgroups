from .gtex import GTEXDataset
from .ace import AceDataset
from .rosmap_singlecell import RosmapSingleCellDataset
from .ace_plasma_csf import AceDatasetPlasmaCSF

def gtex() -> GTEXDataset:
    """
    Create and return an instance of the GTEXDataset class with predefined file paths.

    Returns
    -------
    GTEXDataset
        An instance of the GTEXDataset class initialized with specific data and metadata file paths.
    """
    return GTEXDataset(
        path_to_data='/Users/djuna/Documents/subgroups_data/gtex/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct',
        path_to_meta_data='/Users/djuna/Documents/subgroups_data/gtex/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt',
        path_to_sample_metadata='/Users/djuna/Documents/subgroups_data/gtex/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt',
        n_components=5
    )

def gtex_subset() -> GTEXDataset:
    
    return GTEXDataset(
        path_to_data="/orcd/data/lhtsai/001/djuna/data/gtex_subset/subset_esophagus_bloodvessel.gct",
        path_to_meta_data='/orcd/data/lhtsai/001/djuna/data/gtex_subset/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt',
        path_to_sample_metadata='/orcd/data/lhtsai/001/djuna/data/gtex_subset/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt',
        predicted_class='Esophagus',
        n_components=500
    )

def ace_csf_proteomics() -> AceDataset:
    return AceDataset(
        path_to_data = '/home/Genomica/03-Collabs/djuna/data/202112_Somascan_harpone_db_CSF_ACE_n1370.txt',
        path_to_sample_meta_data = '/home/Genomica/03-Collabs/djuna/data/202406_shared_clinicaldb_CSF_ACE_n1370.txt'
    )

def ace_plasma_proteomics() -> AceDataset:
    return AceDataset(
        path_to_data = '/home/Genomica/03-Collabs/djuna/data/202112_Somascan_harpone_db_CSF_ACE_n1370.txt',
        path_to_sample_meta_data = '/home/Genomica/03-Collabs/djuna/data/202406_shared_clinicaldb_CSF_ACE_n1370.txt',
        path_to_feature_meta_data = '/home/Genomica/03-Collabs/djuna/data/HARPONE-Somalogic_Plasma_Annotations_anmlSMP.xlsx'
    )

def ace_plasma_csf_proteomics() -> AceDatasetPlasmaCSF:
    return AceDatasetPlasmaCSF(
        path_to_data = '/home/Genomica/03-Collabs/djuna/data/202112_Somascan_harpone_db_CSF_ACE_n1370.txt',
        path_to_sample_meta_data = '/home/Genomica/03-Collabs/djuna/data/202406_shared_clinicaldb_CSF_ACE_n1370.txt',
        path_to_feature_meta_data_csf = '/home/Genomica/03-Collabs/djuna/data/HARPONE-Somalogic_CSF_Annotations_anmlSMP.xlsx',
        path_to_feature_meta_data_plasma = '/home/Genomica/03-Collabs/djuna/data/HARPONE-Somalogic_Plasma_Annotations_anmlSMP.xlsx',
        n_components=500
    )

def rosmap_singlecell() -> RosmapSingleCellDataset:
    return RosmapSingleCellDataset(
        path_to_data = '/orcd/data/lhtsai/001/djuna/data/rosmap_mathys400/normalized_batch_corrected_all_celltypes.npy',
        path_to_meta_data = '/orcd/data/lhtsai/001/djuna/data/rosmap_mathys400/meta.npy',
        path_to_extended_meta_data = '/orcd/data/lhtsai/001/djuna/data/rosmap_mathys400/dataset_1282_06-16-2023_long_and_basic_merged_427patients_DianePathAnno_Sorted_DC_2024_06_16.csv',
        path_to_projid_dictionary = '/orcd/data/lhtsai/001/djuna/data/rosmap_mathys400/variable_encodings/projid_dictionary.npy'
    )