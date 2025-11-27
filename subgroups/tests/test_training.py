from ..samplers.mask_generators import mask_factory_fixed_alpha
from ..datasets.test_data import RandomDataset
from ..models.xgboost import model_factory_xgboost
from ..storage.experiment import Experiment
from ..pipelines.pipeline_tc import pipeline_tc
from ..samplers.random_generators import RandomGeneratorTC
import numpy as np
from ..samplers.feature_selectors import select_features_basic
from functools import partial

def test_tc_pipeline_across_seeds():
    """
    Test that the tc pipeline returns different masks and margins for different batch starter seeds.
    """
    random_dataset = RandomDataset()

    exp = Experiment(dataset=random_dataset,
                     mask_factory=partial(mask_factory_fixed_alpha, alpha=0.1),
                     model_factory=model_factory_xgboost,
                     tc_random_generator=RandomGeneratorTC,
                     n_features=5,
                     feature_selector=select_features_basic)

    run1 = pipeline_tc(exp, batch_size=3, batch_starter_seed=1)
    run2 = pipeline_tc(exp, batch_size=3, batch_starter_seed=1)

    assert np.all(run1.masks==run2.masks)
    assert np.all(run1.margins==run2.margins)

    run2 = pipeline_tc(exp, batch_size=3, batch_starter_seed=2)

    assert not np.all(run1.masks==run2.masks)
    assert not np.all(run1.margins==run2.margins)