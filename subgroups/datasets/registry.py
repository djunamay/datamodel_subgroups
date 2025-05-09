from .gtex import GTEXDataset

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
