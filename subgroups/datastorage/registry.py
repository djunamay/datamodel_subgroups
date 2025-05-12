from ..datasets.registry import gtex
from ..datasamplers.mask_generators import fixed_alpha_mask_factory
from ..models.classifier import XgbFactory
from .experiment import Experiment

def gtex_experiment() -> Experiment:
    return Experiment(
        dataset=gtex(),
        mask_factory=fixed_alpha_mask_factory(alpha=0.1),
        model_factory=XgbFactory(),
        in_memory=True,
        n_train_splits=20,
        n_model_inits=15,
        mask_seed=1,
        path="./results/",
        experiment_name="gtex_experiment"
    )
