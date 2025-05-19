from ..datasets.registry import gtex, gtex_subset
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
from ..datamodels.regressor import LassoFactory
from ..datamodels.indices import SequentialIndices
import os

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

def gtex_subset_experiment() -> Experiment:
    path = "./results/"
    name = "gtex_subset_experiment"
    try:
        parameters, alpha = return_best_model_architecture(os.path.join(path, name, "snr_outputs"))
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
        snr_n_passes=10,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path=path,
        experiment_name=name,
        stopping_condition=SNRPrecisionStopping(tolerance=0.1),
        indices_to_fit=SequentialIndices(batch_size=50),
        datamodels_pipeline=DatamodelsPipelineBasic(n_train=9000, 
                                                    datamodel_factory=LassoFactory(),
                                                    path_to_inputs=os.path.join(path, name, "classifier_outputs")),
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