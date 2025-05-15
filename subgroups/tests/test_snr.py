from ..datastorage.mask_margin import MaskMarginStorage
from ..datasamplers.mask_generators import fixed_alpha_mask_factory
from ..datasets.test_data import RandomDataset
from ..models.classifier import XgbFactory, XgbFactoryInitializer
from ..datastorage.experiment import Experiment
from ..experiments.pipeline_tc import pipeline_tc
from ..experiments.compute_signal_to_noise import _mk_train_args, _mk_snr_args
from ..experiments.compute_signal_to_noise import run_training_batch
from ..experiments.pipeline_snr import pipeline_snr
from ..experiments.compute_signal_to_noise import ComputeSNRArgsMultipleArchitectures
from ..datasamplers.mask_generators import fixed_alpha_mask_factory_initializer
from ..datasamplers.random_generators import RandomGeneratorSNR, RandomGeneratorTC
from ..experiments.stopping_condition import SNRPrecisionStopping
import numpy as np

def test_compute_snr_for_one_architecture():
        
    random_dataset = RandomDataset()

    experiment = Experiment(dataset=random_dataset, 
            mask_factory=fixed_alpha_mask_factory(alpha=0.1), 
            model_factory=XgbFactory(), 
            model_factory_initializer=XgbFactoryInitializer(), 
            mask_factory_initializer=fixed_alpha_mask_factory_initializer(),
            in_memory=True, 
            snr_n_models=2, 
            snr_n_passes=2,
            snr_random_generator=RandomGeneratorSNR,
            tc_random_generator=RandomGeneratorTC,
            stopping_condition=SNRPrecisionStopping(tolerance=0.05))

    batch_size = 10

    random_generator = experiment.snr_random_generator(batch_starter_seed=0)

    args = ComputeSNRArgsMultipleArchitectures(dataset=experiment.dataset, 
                                in_memory=experiment.in_memory, 
                                n_models=experiment.snr_n_models, 
                                n_passes=experiment.snr_n_passes, 
                                random_generator=random_generator,
                                path_to_results=experiment.path_to_snr_outputs if not experiment.in_memory else None,
                                n_architectures=batch_size,
                                model_factory_initializer=experiment.model_factory_initializer,
                                mask_factory_initializer=experiment.mask_factory_initializer,
                                stopping_condition=experiment.stopping_condition) 

    new_mask_factory = experiment.mask_factory_initializer.build_mask_factory(random_generator.mask_factory_seed)
    new_model_factory = experiment.model_factory_initializer.build_model_factory(random_generator.model_factory_seed)

    snr_args = _mk_snr_args(args, new_mask_factory, new_model_factory)
    train_args = _mk_train_args(snr_args)
    storage1 = run_training_batch(train_args)
    storage2 = run_training_batch(train_args)

    # verify that the masks are the same across training batches, which is required for the SNR computation
    assert np.array_equal(storage1.masks, storage2.masks)

    assert not np.array_equal(storage1.margins, storage2.margins)

