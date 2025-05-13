from ..datasets.registry import gtex, gtex_subset
from ..datasets.test_data import RandomDataset
from ..datasamplers.mask_generators import fixed_alpha_mask_factory
from ..models.classifier import XgbFactory
from ..models.classifier import XgbFactoryInitializer
from ..datasamplers.mask_generators import fixed_alpha_mask_factory_initializer
from .experiment import Experiment
from ..datasamplers.random_generators import RandomGeneratorSNR, RandomGeneratorTC

def gtex_experiment() -> Experiment:
    return Experiment(
        dataset=gtex(),
        mask_factory=fixed_alpha_mask_factory(alpha=0.1),
        model_factory=XgbFactory(),
        in_memory=True,
        snr_n_models=20,
        snr_n_passes=15,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path="./results/",
        experiment_name="gtex_experiment"
    )

def gtex_subset_experiment() -> Experiment:
    return Experiment(
        dataset=gtex_subset(),
        mask_factory=fixed_alpha_mask_factory(alpha=0.01),
        model_factory=XgbFactory(),
        model_factory_initializer=XgbFactoryInitializer(), 
        mask_factory_initializer=fixed_alpha_mask_factory_initializer(upper_bound=0.2),
        in_memory=False,
        snr_n_models=50,
        snr_n_passes=50,
        snr_random_generator=RandomGeneratorSNR, 
        tc_random_generator=RandomGeneratorTC,
        path="./results/",
        experiment_name="gtex_subset_experiment"
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