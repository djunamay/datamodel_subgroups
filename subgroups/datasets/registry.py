from .gtex import GTEXDataset
from .ace import AceDataset

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
        path_to_sample_metadata='/Users/djuna/Documents/subgroups_data/gtex/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt'
    )

def gtex_subset() -> GTEXDataset:
    
    return GTEXDataset(
        path_to_data="/Users/djuna/Documents/subgroups_data/gtex_subset/subset_esophagus_bloodvessel.gct",
        path_to_meta_data='/Users/djuna/Documents/subgroups_data/gtex/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt',
        path_to_sample_metadata='/Users/djuna/Documents/subgroups_data/gtex/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt',
        predicted_class='Esophagus',
        n_components=5
    )

def ace_csf_proteomics() -> AceDataset:
    return AceDataset(
        path_to_data = '/home/Genomica/03-Collabs/djuna/data/202112_Somascan_harpone_db_CSF_ACE_n1370.txt',
        path_to_meta_data = '/home/Genomica/03-Collabs/djuna/data/202406_shared_clinicaldb_CSF_ACE_n1370.txt'
    )