from ..datasets.registry import gtex, gtex_subset, ace_csf_proteomics, ace_plasma_proteomics, rosmap_singlecell, ace_plasma_csf_proteomics
from ..datasets.test_data import RandomDataset
from ..datasamplers.mask_generators import fixed_alpha_mask_factory
from ..models.classifier import XgbFactory
from ..models.classifier import XgbFactoryInitializer
from ..datasamplers.mask_generators import fixed_alpha_mask_factory_initializer
from .experiment import Experiment
from ..datasamplers.random_generators import RandomGeneratorSNR, RandomGeneratorTC
from ..experiments.stopping_condition import SNRPrecisionStopping
from ..utils.pick_best_architecture import return_best_model_architecture
from ..datamodels.datamodels_pipeline import DatamodelsPipelineBasic
from ..datamodels.regressor import LassoFactory, LinearRegressionFactory
from ..datamodels.indices import SequentialIndices
import os
from ..datasets.ace import AceDataset
from ..datasets.gtex import GTEXDataset
import chz
from .datastorage.combined_mask_margin import CombinedMaskMarginStorage

# def instance_to_class(instance, recursive: bool = True):
#     class NewClass(type(instance)):
#         ...
    
#     for name, field in chz.chz_fields(instance).items():
#         default_value = getattr(instance, name)
#         if chz.is_chz(default_value) and recursive:
#             setattr(NewClass, name, chz.field(default_factory=instance_to_class(default_value, recursive=True)))
#         else:
#             setattr(NewClass, name, chz.field(default=default_value))
#         NewClass.__annotations__[name] = field.final_type
#     return chz.chz(NewClass)

# def overridable(fn):
#     def wrapped():
#         return instance_to_class(fn(), recursive=True)
#     return wrapped()

def gtex_experiment() -> Experiment:
    return Experiment(
        dataset=gtex(),
        mask_factory=fixed_alpha_mask_factory(alpha=0.1),
        model_factory=XgbFactory(),
        in_memory=True,
        snr_n_models=200,
        snr_n_passes=15,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path="./results/",
        experiment_name="gtex_experiment"
    )

def gtex_subset_experiment_home() -> Experiment:
    path = "/Users/djuna/Documents/temp/results/"
    name = "gtex_subset_experiment_home"
    try:
        parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"), acc_cutoff=0)
        mask_factory = fixed_alpha_mask_factory(**alpha)
        model_factory = XgbFactory(**parameters)
    except ValueError:
        mask_factory = fixed_alpha_mask_factory(alpha=0.01)
        model_factory = XgbFactory()

    return Experiment(
        dataset=GTEXDataset(
        path_to_data="../subgroups_data/gtex_subset/subset_esophagus_bloodvessel.gct",
        path_to_meta_data='../subgroups_data/gtex/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt',
        path_to_sample_metadata='../subgroups_data/gtex/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt',
        predicted_class='Esophagus',
        n_components=5
    ),
        mask_factory=mask_factory,
        model_factory=model_factory,
        model_factory_initializer=XgbFactoryInitializer(), 
        mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.2),
        in_memory=False,
        snr_n_models=1000,
        snr_n_passes=50,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path=path,
        experiment_name=name,
        stopping_condition=SNRPrecisionStopping(tolerance=0.1),
        indices_to_fit=SequentialIndices(batch_size=50),
        dm_n_train=9000,
        dm_n_test=1000,
        datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
                                                    path_to_inputs=os.path.join(path, name, "classifier_outputs"),
                                                    path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
    )

def gtex_subset_experiment() -> Experiment:
    path = "/orcd/data/lhtsai/001/djuna/results/"
    name = "gtex_subset_experiment"
    try:
        parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"), acc_cutoff=0)
        mask_factory = fixed_alpha_mask_factory(**alpha)
        model_factory = XgbFactory(**parameters)
    except ValueError:
        mask_factory = fixed_alpha_mask_factory(alpha=0.01)
        model_factory = XgbFactory()

    return Experiment(
        dataset=gtex_subset(),
        mask_factory=mask_factory,
        model_factory=model_factory,
        model_factory_initializer=XgbFactoryInitializer(), 
        mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.2),
        in_memory=False,
        snr_n_models=1000,
        snr_n_passes=50,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path=path,
        experiment_name=name,
        stopping_condition=SNRPrecisionStopping(tolerance=0.1),
        indices_to_fit=SequentialIndices(batch_size=50),
        dm_n_train=9000,
        dm_n_test=1000,
        datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
                                                    combined_mask_margin_storage=CombinedMaskMarginStorage(path_to_inputs=os.path.join(path, name, "classifier_outputs")),
                                                    path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
    )

def random_dataset_experiment() -> Experiment:
    return Experiment(dataset=RandomDataset(), 
           mask_factory=fixed_alpha_mask_factory(alpha=0.1), 
           model_factory=XgbFactory(), 
           model_factory_initializer=XgbFactoryInitializer(), 
           mask_factory_initializer=fixed_alpha_mask_factory_initializer(),
           in_memory=True, 
           snr_n_models=20, 
           snr_n_passes=3, 
           snr_random_generator=RandomGeneratorSNR, 
           tc_random_generator=RandomGeneratorTC,
           )

#@overridable # TODO: have to fix this / move each function into own file because overridable gets called on import of any function in this file, meaning that it raises an error when the data isnt found
def ace_csf_proteomics_amnestic_experiment() -> Experiment: # TODO: The overwrite config doesn't work well when running the snr pipeline as independent batches with different seeds - Need to set overwrite to True then since the best model architecture can change over time. Fix this config issue.
    # TODO: overridable works but not recurisvely and means that the config file is not well-documented
    path = "/home/Genomica/03-Collabs/djuna/results/"
    name = "ace_csf_proteomics_amnestic_experiment"
    try:
        parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"), acc_cutoff=0)
        mask_factory = fixed_alpha_mask_factory(**alpha)
        model_factory = XgbFactory(**parameters)
    except ValueError:
        mask_factory = fixed_alpha_mask_factory(alpha=0.01)
        model_factory = XgbFactory()

    return Experiment(
        dataset=ace_csf_proteomics(),
        mask_factory=mask_factory,
        model_factory=model_factory,
        model_factory_initializer=XgbFactoryInitializer(), 
        mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.3), # this upper bound ensures at maximum 70% sampling of the smaller class for training
        in_memory=False,
        snr_n_models=1000,
        snr_n_passes=50,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path=path,
        experiment_name=name,
        stopping_condition=SNRPrecisionStopping(tolerance=0.1),
        indices_to_fit=SequentialIndices(batch_size=1171),
        dm_n_train=500000,
        dm_n_test=500000,
        datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
                                                    path_to_inputs=os.path.join(path, name, "classifier_outputs"),
                                                    path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
        notes="This experiment was performed after implementing additional filtering of the data to remove features that were not human or that had failed the column check, as well as MAPT or APP."
    )

def ace_plasma_proteomics_amnestic_experiment() -> Experiment: # TODO: The overwrite config doesn't work well when running the snr pipeline as independent batches with different seeds - Need to set overwrite to True then since the best model architecture can change over time. Fix this config issue.
    # TODO: overridable works but not recurisvely and means that the config file is not well-documented
    path = "/home/Genomica/03-Collabs/djuna/results/"
    name = "ace_plasma_proteomics_amnestic_experiment"
    try:
        parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"), acc_cutoff=0)
        mask_factory = fixed_alpha_mask_factory(**alpha)
        model_factory = XgbFactory(**parameters)
    except ValueError:
        mask_factory = fixed_alpha_mask_factory(alpha=0.01)
        model_factory = XgbFactory()

    return Experiment(
        dataset=ace_plasma_proteomics(),
        mask_factory=mask_factory,
        model_factory=model_factory,
        model_factory_initializer=XgbFactoryInitializer(), 
        mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.3), # this upper bound ensures at maximum 70% sampling of the smaller class for training
        in_memory=False,
        snr_n_models=1000,
        snr_n_passes=50,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path=path,
        experiment_name=name,
        stopping_condition=SNRPrecisionStopping(tolerance=0.1),
        indices_to_fit=SequentialIndices(batch_size=1171),
        dm_n_train=500000,
        dm_n_test=500000,
        datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
                                                    path_to_inputs=os.path.join(path, name, "classifier_outputs"),
                                                    path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
        notes="This experiment was performed after implementing additional filtering of the data to remove features that were not human or that had failed the column check, as well as MAPT or APP."
    )

def ace_plasma_csf_proteomics_amnestic_experiment() -> Experiment: # TODO: The overwrite config doesn't work well when running the snr pipeline as independent batches with different seeds - Need to set overwrite to True then since the best model architecture can change over time. Fix this config issue.
    # TODO: overridable works but not recurisvely and means that the config file is not well-documented
    path = "/home/Genomica/03-Collabs/djuna/results/"
    name = "ace_plasma__csf_proteomics_amnestic_experiment"
    try:
        parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"), acc_cutoff=0)
        mask_factory = fixed_alpha_mask_factory(**alpha)
        model_factory = XgbFactory(**parameters)
    except ValueError:
        mask_factory = fixed_alpha_mask_factory(alpha=0.01)
        model_factory = XgbFactory()

    return Experiment(
        dataset=ace_plasma_csf_proteomics(),
        mask_factory=mask_factory,
        model_factory=model_factory,
        model_factory_initializer=XgbFactoryInitializer(), 
        mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.3), # this upper bound ensures at maximum 70% sampling of the smaller class for training
        in_memory=False,
        snr_n_models=1000,
        snr_n_passes=50,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path=path,
        experiment_name=name,
        stopping_condition=SNRPrecisionStopping(tolerance=0.1),
        indices_to_fit=SequentialIndices(batch_size=1171),
        dm_n_train=500000,
        dm_n_test=500000,
        datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
                                                    path_to_inputs=os.path.join(path, name, "classifier_outputs"),
                                                    path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
        notes="This experiment was performed after implementing additional filtering of the data to remove features that were not human or that had failed the column check, as well as MAPT or APP."
    )

def ace_csf_proteomics_age_experiment() -> Experiment: # TODO: The overwrite config doesn't work well when running the snr pipeline as independent batches with different seeds - Need to set overwrite to True then since the best model architecture can change over time. Fix this config issue.
    # TODO: overridable works but not recurisvely and means that the config file is not well-documented
    path = "/home/Genomica/03-Collabs/djuna/results/"
    name = "ace_csf_proteomics_age_experiment"
    try:
        parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"), acc_cutoff=0)
        mask_factory = fixed_alpha_mask_factory(**alpha)
        model_factory = XgbFactory(**parameters)
    except ValueError:
        mask_factory = fixed_alpha_mask_factory(alpha=0.01)
        model_factory = XgbFactory()

    return Experiment(
        dataset=AceDataset(
        path_to_data = '/home/Genomica/03-Collabs/djuna/data/202112_Somascan_harpone_db_CSF_ACE_n1370.txt',
        path_to_sample_meta_data = '/home/Genomica/03-Collabs/djuna/data/202406_shared_clinicaldb_CSF_ACE_n1370.txt',
        split='age_group'),
        mask_factory=mask_factory,
        model_factory=model_factory,
        model_factory_initializer=XgbFactoryInitializer(), 
        mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.7), # this upper bound ensures at maximum 70% sampling of the smaller class for training
        in_memory=False,
        snr_n_models=1000,
        snr_n_passes=50,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path=path,
        experiment_name=name,
        stopping_condition=SNRPrecisionStopping(tolerance=0.1),
        indices_to_fit=SequentialIndices(batch_size=1171),
        dm_n_train=500000,
        dm_n_test=500000,
        datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
                                                    path_to_inputs=os.path.join(path, name, "classifier_outputs"),
                                                    path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
        notes="This experiment was performed after implementing additional filtering of the data to remove features that were not human or that had failed the column check."
    )

def rosmap_singlecell_experiment() -> Experiment: # TODO: The overwrite config doesn't work well when running the snr pipeline as independent batches with different seeds - Need to set overwrite to True then since the best model architecture can change over time. Fix this config issue.
    path = "/orcd/data/lhtsai/001/djuna/results/"
    name = "rosmap_singlecell_experiment"
    try:
        parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"), acc_cutoff=0)
        mask_factory = fixed_alpha_mask_factory(**alpha)
        model_factory = XgbFactory(**parameters)
    except ValueError:
        mask_factory = fixed_alpha_mask_factory(alpha=0.01)
        model_factory = XgbFactory()

    return Experiment(
        dataset=rosmap_singlecell(),
        mask_factory=mask_factory,
        model_factory=model_factory,
        model_factory_initializer=XgbFactoryInitializer(), 
        mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.45), # this upper bound ensures at maximum 70% sampling of the smaller class for training
        in_memory=False,
        snr_n_models=1000,
        snr_n_passes=50,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path=path,
        experiment_name=name,
        stopping_condition=SNRPrecisionStopping(tolerance=0.1),
        indices_to_fit=SequentialIndices(batch_size=50),
        dm_n_train=9000,
        dm_n_test=1000,
        datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
                                                    path_to_inputs=os.path.join(path, name, "classifier_outputs"),
                                                    path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
    )
