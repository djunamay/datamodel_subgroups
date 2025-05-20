from ..datastorage.mask_margin import MaskMarginStorage
from ..datasamplers.mask_generators import fixed_alpha_mask_factory
from ..datasets.test_data import RandomDataset
from ..models.classifier import XgbFactory, XgbFactoryInitializer
from ..datastorage.experiment import Experiment
from ..experiments.pipeline_tc import pipeline_tc
from ..experiments.pipeline_snr import pipeline_snr
from ..datasamplers.mask_generators import fixed_alpha_mask_factory_initializer
from ..datasamplers.random_generators import RandomGeneratorSNR, RandomGeneratorTC
from ..experiments.stopping_condition import SNRPrecisionStopping
from ..experiments.compute_signal_to_noise import ComputeSNRArgs, snr_inputs_for_one_architecture, compute_signal_noise
import numpy as np

def test_tc_pipeline_across_seeds():
    """
    Test that the tc pipeline returns different masks and margins for different batch starter seeds.
    """
    random_dataset = RandomDataset()

    exp = Experiment(dataset=random_dataset, 
            mask_factory=fixed_alpha_mask_factory(alpha=0.1), 
            model_factory=XgbFactory(), 
            model_factory_initializer=XgbFactoryInitializer(), 
            mask_factory_initializer=fixed_alpha_mask_factory_initializer(),
            in_memory=True, 
            snr_n_models=5, 
            snr_n_passes=3,
            snr_random_generator=RandomGeneratorSNR,
            tc_random_generator=RandomGeneratorTC,
            stopping_condition=SNRPrecisionStopping(tolerance=0.05))

    run1 = pipeline_tc(exp, batch_size=3, batch_starter_seed=1)
    run2 = pipeline_tc(exp, batch_size=3, batch_starter_seed=1)

    assert np.all(run1.masks==run2.masks)
    assert np.all(run1.margins==run2.margins)

    run2 = pipeline_tc(exp, batch_size=3, batch_starter_seed=2)

    assert not np.all(run1.masks==run2.masks)
    assert not np.all(run1.margins==run2.margins)