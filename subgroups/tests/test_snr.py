from ..pipelines.train_classifiers import run_training_batch, TrainClassifiersArgs
from ..pipelines.compute_signal_to_noise import ComputeSNRArgsMultipleArchitectures
from ..samplers.mask_generators import mask_factory_fixed_alpha
from ..datasets.test_data import RandomDataset
from ..models.xgboost import model_factory_xgboost, model_factory_init_xgboost
from ..storage.experiment import SNRExperiment
from ..pipelines.pipeline_snr import pipeline_snr
from ..samplers.mask_generators import mask_factory_init_fixed_alpha
from ..samplers.random_generators import RandomGeneratorSNR
from ..pipelines.stopping_condition import SNRPrecisionStopping
from ..pipelines.compute_signal_to_noise import ComputeSNRArgs, snr_inputs_for_one_architecture, compute_signal_noise
import numpy as np
from ..utils.random import fork_rng
from ..samplers.feature_selectors import select_features_basic
from functools import partial

def test_fork_rng():
    children = fork_rng(np.random.default_rng(0), 6)
    children2 = fork_rng(np.random.default_rng(0), 6)
    for child1, child2 in zip(children, children2):
        assert child1.random() == child2.random()

    assert len(set([child.random() for child in children])) == 6

def test_compute_snr_for_one_architecture():
    random_dataset = RandomDataset()

    experiment = SNRExperiment(dataset=random_dataset,
                               model_factory_initializer=model_factory_init_xgboost,
                               mask_factory_initializer=mask_factory_init_fixed_alpha,
                               snr_n_models=2,
                               snr_n_passes=2,
                               snr_random_generator=RandomGeneratorSNR,
                               stopping_condition=SNRPrecisionStopping(tolerance=0.05),
                               feature_selector=select_features_basic,
                               npcs_min=2,
                               npcs_max=5)

    batch_size = 10
    random_generator = experiment.snr_random_generator(batch_starter_seed=0)

    args = ComputeSNRArgsMultipleArchitectures(dataset=experiment.dataset,
                        in_memory=True,
                        n_models=experiment.snr_n_models,
                        n_passes=experiment.snr_n_passes, 
                        random_generator=random_generator,
                        path_to_results=experiment.path_to_snr_outputs if not True else None,
                        n_architectures=batch_size,
                        model_factory_initializer=experiment.model_factory_initializer,
                        mask_factory_initializer=experiment.mask_factory_initializer,
                        stopping_condition=experiment.stopping_condition,
                        feature_selector=experiment.feature_selector,
                        npcs_min=experiment.npcs_min,
                        npcs_max=experiment.npcs_max)
    
    def make_storage(random_generator: RandomGeneratorSNR, args: ComputeSNRArgsMultipleArchitectures):
        
        new_mask_factory = experiment.mask_factory_initializer(random_generator.mask_factory_rng)
        new_model_factory = experiment.model_factory_initializer(random_generator.model_factory_rng)

        train_args = TrainClassifiersArgs(dataset=experiment.dataset,
                             mask_factory=new_mask_factory,
                             model_factory=new_model_factory,
                             n_models=10,
                             in_memory=True,
                             path=None,
                             random_generator=random_generator,
                             npcs=3,
                             feature_selector=experiment.feature_selector)
        return run_training_batch(train_args, batch_starter_seed=None)

    storage1 = make_storage(random_generator, args)
    storage2 = make_storage(random_generator, args)

    # verify that each time the random_generator is called, the masks & model are different
    assert not np.array_equal(storage1.masks, storage2.masks)

    assert not np.array_equal(storage1.margins, storage2.margins)


def test_snr_across_seeds():
    """
    Test that the snr pipeline returns different snr values across different batch starter seeds.
    """
    random_dataset = RandomDataset()

    exp = SNRExperiment(dataset=random_dataset,
                        model_factory_initializer=model_factory_init_xgboost,
                        mask_factory_initializer=mask_factory_init_fixed_alpha,
                        snr_n_models=2,
                        snr_n_passes=2,
                        snr_random_generator=RandomGeneratorSNR,
                        stopping_condition=SNRPrecisionStopping(tolerance=0.05),
                        feature_selector=select_features_basic,
                        npcs_max=20)

    snr1 = pipeline_snr(exp, batch_size=2, batch_starter_seed=1)
    snr2 = pipeline_snr(exp, batch_size=2, batch_starter_seed=1)

    assert np.array_equal(snr1, snr2, equal_nan=True)

    snr3 = pipeline_snr(exp, batch_size=2, batch_starter_seed=2)

    assert not np.array_equal(snr1, snr3, equal_nan=True)


def test_consistency_of_snr_computation():
    stopping_condition = SNRPrecisionStopping()

    args = ComputeSNRArgs(
        dataset=RandomDataset(),
        mask_factory=partial(mask_factory_fixed_alpha, alpha=0.01),
        model_factory=model_factory_xgboost,
        random_generator=RandomGeneratorSNR(batch_starter_seed=0),
        stopping_condition=SNRPrecisionStopping(),
        n_models=10,
        n_passes=10,
        feature_selector=select_features_basic
    )

    margins, masks, _, _ = snr_inputs_for_one_architecture(args)

    E_init, V_init = stopping_condition._E_init_V_init(margins, masks)
    snr = stopping_condition._compute_snr(E_init, V_init)
    
    assert np.mean(snr) == np.mean(compute_signal_noise(margins, masks))