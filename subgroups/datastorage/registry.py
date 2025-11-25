from ..datasets.registry import gtex, gtex_subset, ace_csf_proteomics, ace_plasma_proteomics, rosmap_singlecell, ace_plasma_csf_proteomics, gtex_subset_home
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
from .combined_mask_margin import CombinedMaskMarginStorage
from ..datasamplers.feature_selectors import SelectPCsBasic, SelectPCsSingleCell


def gtex_subset_experiment_home() -> Experiment:
    path = "/Users/djuna/Documents/CurrentDocuments/current_projects_code/datamodel_subgroups/results/"
    name = "gtex_subset_experiment_june_30"
    mask_factory = fixed_alpha_mask_factory(alpha=0.012507530044163674)
    model_factory = XgbFactory(max_depth=7)

    return Experiment(
        dataset=gtex_subset_home(),
        mask_factory=mask_factory,
        model_factory=model_factory,
        model_factory_initializer=XgbFactoryInitializer(), 
        mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.2),
        snr_n_models=1000,
        snr_n_passes=50,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path=path,
        experiment_name=name,
        stopping_condition=SNRPrecisionStopping(tolerance=0.1),
        indices_to_fit=SequentialIndices(batch_size=50),
        dm_n_train=1500000,
        dm_n_test=500000,
        npcs_min=5,
        npcs_max=500,
        npcs=20,
        feature_selector=SelectPCsBasic(),
        datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
                                                    combined_mask_margin_storage=CombinedMaskMarginStorage(path_to_inputs=os.path.join(path, name, "classifier_outputs")),
                                                    path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
    )
    
def gtex_subset_experiment() -> Experiment:
    path = "/orcd/data/lhtsai/001/djuna/results/"
    name = "gtex_subset_experiment_june_30"
    # try:
    #     parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"), acc_cutoff=0)
    #     mask_factory = fixed_alpha_mask_factory(**alpha)
    #     model_factory = XgbFactory(**parameters)
    # except ValueError:
    mask_factory = fixed_alpha_mask_factory(alpha=0.012507530044163674)
    model_factory = XgbFactory(max_depth=7)

    return Experiment(
        dataset=gtex_subset(),
        mask_factory=mask_factory,
        model_factory=model_factory,
        model_factory_initializer=XgbFactoryInitializer(), 
        mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.2),
        snr_n_models=1000,
        snr_n_passes=50,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path=path,
        experiment_name=name,
        stopping_condition=SNRPrecisionStopping(tolerance=0.1),
        indices_to_fit=SequentialIndices(batch_size=50),
        dm_n_train=1500000,
        dm_n_test=500000,
        npcs_min=5,
        npcs_max=500,
        npcs=20,
        feature_selector=SelectPCsBasic(),
        #counterfactual_inputs=CounterfactualInputsBasic,
        #counterfactual_estimator=CounterfactualEvaluation,
        datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
                                                    combined_mask_margin_storage=CombinedMaskMarginStorage(path_to_inputs=os.path.join(path, name, "classifier_outputs")),
                                                    path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
    )

def random_dataset_experiment() -> Experiment:
    return Experiment(dataset=RandomDataset(), 
           mask_factory=fixed_alpha_mask_factory(alpha=0.1), 
           model_factory=XgbFactory(),
           tc_random_generator=RandomGeneratorTC,
           feature_selector=SelectPCsBasic(),
           )

#@overridable # TODO: have to fix this / move each function into own file because overridable gets called on import of any function in this file, meaning that it raises an error when the data isnt found
# def ace_csf_proteomics_amnestic_experiment() -> Experiment: # TODO: The overwrite config doesn't work well when running the snr pipeline as independent batches with different seeds - Need to set overwrite to True then since the best model architecture can change over time. Fix this config issue.
#     # TODO: overridable works but not recurisvely and means that the config file is not well-documented
#     path = "/home/Genomica/03-Collabs/djuna/results/"
#     name = "ace_csf_proteomics_amnestic_experiment_june_30"
#     try:
#         parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"), acc_cutoff=0)
#         mask_factory = fixed_alpha_mask_factory(**alpha)
#         model_factory = XgbFactory(**parameters)
#     except ValueError:
#         mask_factory = fixed_alpha_mask_factory(alpha=0.01)
#         model_factory = XgbFactory()

#     return Experiment(
#         dataset=ace_csf_proteomics(),
#         mask_factory=mask_factory,
#         model_factory=model_factory,
#         model_factory_initializer=XgbFactoryInitializer(), 
#         mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.3), # this upper bound ensures at maximum 70% sampling of the smaller class for training
#
#         snr_n_models=1000,
#         snr_n_passes=50,
#         snr_random_generator=RandomGeneratorSNR, 
#         tc_random_generator=RandomGeneratorTC,
#         path=path,
#         experiment_name=name,
#         stopping_condition=SNRPrecisionStopping(tolerance=0.1),
#         indices_to_fit=SequentialIndices(batch_size=1171),
#         dm_n_train=500000,
#         dm_n_test=500000,
#         datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
#                                                     combined_mask_margin_storage=CombinedMaskMarginStorage(path_to_inputs=os.path.join(path, name, "classifier_outputs")),
#                                                     path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
#         notes="This experiment was performed after implementing additional filtering of the data to remove features that were not human or that had failed the column check, as well as MAPT or APP."
#     )

# def ace_plasma_proteomics_amnestic_experiment() -> Experiment: # TODO: The overwrite config doesn't work well when running the snr pipeline as independent batches with different seeds - Need to set overwrite to True then since the best model architecture can change over time. Fix this config issue.
#     # TODO: overridable works but not recurisvely and means that the config file is not well-documented
#     path = "/home/Genomica/03-Collabs/djuna/results/"
#     name = "ace_plasma_proteomics_amnestic_experiment"
#     try:
#         parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"), acc_cutoff=0)
#         mask_factory = fixed_alpha_mask_factory(**alpha)
#         model_factory = XgbFactory(**parameters)
#     except ValueError:
#         mask_factory = fixed_alpha_mask_factory(alpha=0.01)
#         model_factory = XgbFactory()

#     return Experiment(
#         dataset=ace_plasma_proteomics(),
#         mask_factory=mask_factory,
#         model_factory=model_factory,
#         model_factory_initializer=XgbFactoryInitializer(), 
#         mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.3), # this upper bound ensures at maximum 70% sampling of the smaller class for training
#
#         snr_n_models=1000,
#         snr_n_passes=50,
#         snr_random_generator=RandomGeneratorSNR, 
#         tc_random_generator=RandomGeneratorTC,
#         path=path,
#         experiment_name=name,
#         stopping_condition=SNRPrecisionStopping(tolerance=0.1),
#         indices_to_fit=SequentialIndices(batch_size=1171),
#         dm_n_train=500000,
#         dm_n_test=500000,
#         datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
#                                                     combined_mask_margin_storage=CombinedMaskMarginStorage(path_to_inputs=os.path.join(path, name, "classifier_outputs")),
#                                                     path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
#         notes="This experiment was performed after implementing additional filtering of the data to remove features that were not human or that had failed the column check, as well as MAPT or APP."
#     )

def ace_plasma_csf_proteomics_amnestic_experiment() -> Experiment: # TODO: The overwrite config doesn't work well when running the snr pipeline as independent batches with different seeds - Need to set overwrite to True then since the best model architecture can change over time. Fix this config issue.
    # TODO: overridable works but not recurisvely and means that the config file is not well-documented
    path = "/home/Genomica/03-Collabs/djuna/results/"
    name = "ace_plasma_csf_proteomics_amnestic_experiment_june_30"
    # try:
    #     parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"), acc_cutoff=0)
    #     mask_factory = fixed_alpha_mask_factory(**alpha)
    #     model_factory = XgbFactory(**parameters)
    # except ValueError:
    mask_factory = fixed_alpha_mask_factory(alpha=0.13740237552428117)
    model_factory = XgbFactory(max_depth=3)

    return Experiment(
        dataset=ace_plasma_csf_proteomics(),
        mask_factory=mask_factory,
        model_factory=model_factory,
        model_factory_initializer=XgbFactoryInitializer(), 
        mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.3), # this upper bound ensures at maximum 70% sampling of the smaller class for training
        snr_n_models=1000,
        snr_n_passes=50,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path=path,
        experiment_name=name,
        stopping_condition=SNRPrecisionStopping(tolerance=0.1),
        indices_to_fit=SequentialIndices(batch_size=50),
        dm_n_train=1000000,
        dm_n_test=500000,
        npcs_min=5,
        npcs_max=500,
        npcs=47,
        feature_selector=SelectPCsBasic(),
        #counterfactual_inputs=CounterfactualInputsSingleCell,
        #counterfactual_estimator=CounterfactualEvaluation,
        datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
                                                    combined_mask_margin_storage=CombinedMaskMarginStorage(path_to_inputs=os.path.join(path, name, "classifier_outputs")),
                                                    path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
        notes="This experiment was performed after implementing additional filtering of the data to remove features that were not human or that had failed the column check, as well as MAPT or APP."
    )

# def ace_csf_proteomics_age_experiment() -> Experiment: # TODO: The overwrite config doesn't work well when running the snr pipeline as independent batches with different seeds - Need to set overwrite to True then since the best model architecture can change over time. Fix this config issue.
#     # TODO: overridable works but not recurisvely and means that the config file is not well-documented
#     path = "/home/Genomica/03-Collabs/djuna/results/"
#     name = "ace_csf_proteomics_age_experiment"
#     try:
#         parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"), acc_cutoff=0)
#         mask_factory = fixed_alpha_mask_factory(**alpha)
#         model_factory = XgbFactory(**parameters)
#     except ValueError:
#         mask_factory = fixed_alpha_mask_factory(alpha=0.01)
#         model_factory = XgbFactory()

#     return Experiment(
#         dataset=AceDataset(
#         path_to_data = '/home/Genomica/03-Collabs/djuna/data/202112_Somascan_harpone_db_CSF_ACE_n1370.txt',
#         path_to_sample_meta_data = '/home/Genomica/03-Collabs/djuna/data/202406_shared_clinicaldb_CSF_ACE_n1370.txt',
#         split='age_group'),
#         mask_factory=mask_factory,
#         model_factory=model_factory,
#         model_factory_initializer=XgbFactoryInitializer(), 
#         mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.7), # this upper bound ensures at maximum 70% sampling of the smaller class for training
#
#         snr_n_models=1000,
#         snr_n_passes=50,
#         snr_random_generator=RandomGeneratorSNR, 
#         tc_random_generator=RandomGeneratorTC,
#         path=path,
#         experiment_name=name,
#         stopping_condition=SNRPrecisionStopping(tolerance=0.1),
#         indices_to_fit=SequentialIndices(batch_size=1171),
#         dm_n_train=500000,
#         dm_n_test=500000,
#         datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
#                                                     combined_mask_margin_storage=CombinedMaskMarginStorage(path_to_inputs=os.path.join(path, name, "classifier_outputs")),
#                                                     path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
#         notes="This experiment was performed after implementing additional filtering of the data to remove features that were not human or that had failed the column check."
#     )

def rosmap_singlecell_experiment() -> Experiment: # TODO: The overwrite config doesn't work well when running the snr pipeline as independent batches with different seeds - Need to set overwrite to True then since the best model architecture can change over time. Fix this config issue.
    path = "/orcd/data/lhtsai/001/djuna/results/"
    name = "rosmap_singlecell_experiment_june_30"
    ## POINT 2
    # try:
    #     parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"), acc_cutoff=0)
    #     mask_factory = fixed_alpha_mask_factory(**alpha)
    #     model_factory = XgbFactory(**parameters)
    # except ValueError:
    mask_factory = fixed_alpha_mask_factory(alpha=0.12736504486974448)
    model_factory = XgbFactory(max_depth=10)

    return Experiment(
        dataset=rosmap_singlecell(),
        mask_factory=mask_factory,
        model_factory=model_factory,
        model_factory_initializer=XgbFactoryInitializer(), 
        mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.45), # this upper bound ensures at maximum 70% sampling of the smaller class for training
        snr_n_models=1000,
        snr_n_passes=50,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path=path,
        experiment_name=name,
        stopping_condition=SNRPrecisionStopping(tolerance=0.1),
        indices_to_fit=SequentialIndices(batch_size=421),
        dm_n_train=3000000,
        dm_n_test=500000,
        npcs_min=5,
        npcs_max=50,
        npcs=5,
        feature_selector=SelectPCsSingleCell(),
        counterfactual_test_fraction=0.1,
        #counterfactual_inputs=CounterfactualInputsBasic,
        #counterfactual_estimator=CounterfactualEvaluation,
        datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
                                                    combined_mask_margin_storage=CombinedMaskMarginStorage(path_to_inputs=os.path.join(path, name, "classifier_outputs")),
                                                    path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
    )

def rosmap_singlecell_experiment_point_1() -> Experiment: # TODO: The overwrite config doesn't work well when running the snr pipeline as independent batches with different seeds - Need to set overwrite to True then since the best model architecture can change over time. Fix this config issue.
    path = "/orcd/data/lhtsai/001/djuna/results/"
    name = "rosmap_singlecell_experiment_june_30_point_1"
    mask_factory = fixed_alpha_mask_factory(alpha=0.027943688549876573)
    model_factory = XgbFactory(max_depth=4)

    return Experiment(
        dataset=rosmap_singlecell(),
        mask_factory=mask_factory,
        model_factory=model_factory,
        model_factory_initializer=XgbFactoryInitializer(), 
        mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.45), # this upper bound ensures at maximum 70% sampling of the smaller class for training
        snr_n_models=1000,
        snr_n_passes=50,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path=path,
        experiment_name=name,
        stopping_condition=SNRPrecisionStopping(tolerance=0.1),
        indices_to_fit=SequentialIndices(batch_size=421),
        dm_n_train=3000000,
        dm_n_test=500000,
        npcs_min=5,
        npcs_max=50,
        npcs=40,
        feature_selector=SelectPCsSingleCell(),
        datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
                                                    combined_mask_margin_storage=CombinedMaskMarginStorage(path_to_inputs=os.path.join(path, name, "classifier_outputs")),
                                                    path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
    )

def rosmap_singlecell_experiment_point_3() -> Experiment: # TODO: The overwrite config doesn't work well when running the snr pipeline as independent batches with different seeds - Need to set overwrite to True then since the best model architecture can change over time. Fix this config issue.
    path = "/orcd/data/lhtsai/001/djuna/results/"
    name = "rosmap_singlecell_experiment_june_30_point_3"
    # try:
    #     parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"), acc_cutoff=0)
    #     mask_factory = fixed_alpha_mask_factory(**alpha)
    #     model_factory = XgbFactory(**parameters)
    # except ValueError:
    mask_factory = fixed_alpha_mask_factory(alpha=0.3190159948034689)
    model_factory = XgbFactory(max_depth=7)

    return Experiment(
        dataset=rosmap_singlecell(),
        mask_factory=mask_factory,
        model_factory=model_factory,
        model_factory_initializer=XgbFactoryInitializer(), 
        mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.45), # this upper bound ensures at maximum 70% sampling of the smaller class for training
        snr_n_models=1000,
        snr_n_passes=50,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path=path,
        experiment_name=name,
        stopping_condition=SNRPrecisionStopping(tolerance=0.1),
        indices_to_fit=SequentialIndices(batch_size=421),
        dm_n_train=3000000,
        dm_n_test=500000,
        npcs_min=5,
        npcs_max=50,
        npcs=5,
        feature_selector=SelectPCsSingleCell(),
        #counterfactual_inputs=CounterfactualInputsSingleCell,
        #counterfactual_estimator=CounterfactualEvaluation,
        datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
                                                    combined_mask_margin_storage=CombinedMaskMarginStorage(path_to_inputs=os.path.join(path, name, "classifier_outputs")),
                                                    path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
    )



# def rosmap_singlecell_experiment_baseline() -> Experiment: # TODO: The overwrite config doesn't work well when running the snr pipeline as independent batches with different seeds - Need to set overwrite to True then since the best model architecture can change over time. Fix this config issue.
#     path = "/orcd/data/lhtsai/001/djuna/results/"
#     name = "rosmap_singlecell_experiment_baseline"
#     xgb_params = {'learning_rate':0.1666, 'max_depth':2, 'n_estimators':133, 'reg_lambda':1.5, 'reg_alpha':0.5, 'subsample':0.25}

#     mask_factory = fixed_alpha_mask_factory(alpha=0.25)
#     model_factory = XgbFactory(**xgb_params)

#     return Experiment(
#         dataset=rosmap_singlecell(),
#         mask_factory=mask_factory,
#         model_factory=model_factory,
#         model_factory_initializer=XgbFactoryInitializer(), 
#         mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.45), # this upper bound ensures at maximum 70% sampling of the smaller class for training
#
#         snr_n_models=1000,
#         snr_n_passes=50,
#         snr_random_generator=RandomGeneratorSNR, 
#         tc_random_generator=RandomGeneratorTC,
#         path=path,
#         experiment_name=name,
#         stopping_condition=SNRPrecisionStopping(tolerance=0.1),
#         indices_to_fit=SequentialIndices(batch_size=421),
#         dm_n_train=3000000,
#         dm_n_test=500000,
#         datamodels_pipeline=DatamodelsPipelineBasic(datamodel_factory=LinearRegressionFactory(),
#                                                     combined_mask_margin_storage=CombinedMaskMarginStorage(path_to_inputs=os.path.join(path, name, "classifier_outputs")),
#                                                     path_to_outputs=os.path.join(path, name, "datamodel_outputs")),
#     )
