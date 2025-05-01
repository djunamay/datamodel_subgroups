from .gtex import GTEXDataset
# make absolute path
def gtex():
    return GTEXDataset(path_to_data='/Users/djuna/Documents/subgroups_data/gtex/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct', 
                            path_to_meta_data='/Users/djuna/Documents/subgroups_data/gtex/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt', 
                            path_to_sample_metadata='/Users/djuna/Documents/subgroups_data/gtex/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt')
