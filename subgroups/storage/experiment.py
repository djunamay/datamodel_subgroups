import os
from pathlib import Path
from typing import Type

import chz

from subgroups.models.regressor import datamodel_factory_linear, datamodel_factory_fn
from ..samplers import mask_factory_fn, mask_factory_init_fn
from ..samplers.feature_selectors import select_features_basic, select_features_fn
from ..samplers.mask_generators import mask_factory_fixed_alpha
from ..samplers.random_generators import (
    RandomGeneratorSNR,
    RandomGeneratorTC,
)
from ..datasets import DatasetInterface
from ..pipelines.stopping_condition import StoppingConditionInterface
from ..models import model_factory_fn, model_factory_init_fn
from ..models.xgboost import model_factory_xgboost


@chz.chz
class ExperimentBase:
    """
    Class containing all the necessary information for running an experiment.

    Attributes
    ----------
    dataset : DatasetInterface
        The dataset used for the experiment.
    experiment_name : str
        Name of the experiment.
    path : Path
        Path to store experiment results if not in memory.
    notes : str
        Notes for the experiment.
    feature_selector : select_features_fn
        Feature selector for the experiment.
    """

    dataset: DatasetInterface = chz.field(doc='The dataset used for the experiment.')
    path: str = chz.field(default='.', doc='Path to store experiment results if not in memory.')
    experiment_name: str = chz.field(default='NoName', doc='Name of the experiment if not in memory.')
    notes: str=chz.field(default=None, doc='Notes for the experiment.')
    feature_selector: select_features_fn = chz.field(default=select_features_basic, doc='Feature selector for the experiment.')

    @chz.init_property
    def _check_path(self):
        """
        Check if the path to the experiment results is provided if in_memory is False.
        """
        if self.path is None:
            raise ValueError("path to output must be provided if in_memory is False")
        if self.path is not None and self.experiment_name is None:
            raise ValueError("experiment_name must be provided if path is provided")

    @chz.init_property
    def path_to_results(self):
        """
        Path to the results of the experiment.
        """
        if self.path is None:
            return None
        else:
            return os.path.join(self.path, self.experiment_name)

    @chz.init_property
    def path_to_classifier_outputs(self):
        """
        Path to the models outputs of the experiment.
        """
        if self.path is None:
            return None
        else:
            return os.path.join(self.path_to_results, 'classifier_outputs')

    @chz.init_property
    def path_to_datamodel_outputs(self):
        """
        Path to the datamodel outputs of the experiment.
        """
        if self.path is None:
            return None
        else:
            return os.path.join(self.path_to_results, 'datamodel_outputs')

    @chz.init_property
    def path_to_clustering_outputs(self):
        """
        Path to the clustering outputs of the experiment.
        """
        if self.path is None:
            return None
        else:
            return os.path.join(self.path_to_results, 'clustering_outputs')

    @chz.init_property
    def path_to_benchmarks(self):
        """
        Path to the benchmarks of the experiment.
        """
        if self.path is None:
            return None
        else:
            return os.path.join(self.path_to_results, 'benchmarks')

    @chz.init_property
    def path_to_plots(self):
        """
        Path to the plots of the experiment.
        """
        if self.path is None:
            return None
        else:
            return os.path.join(self.path_to_results, 'plots')

    @chz.init_property
    def path_to_snr_outputs(self):
        """
        Path to the SNR outputs of the experiment.
        """
        if self.path is None:
            return None
        else:
            return os.path.join(self.path_to_results, 'snr_outputs')

    @chz.init_property
    def _prepare_experiment(self):
        """
        If not in memory, prepare an experiment with the given configuration.
        """
        if not self.path is None:
            if not os.path.exists(self.path_to_results):
                os.makedirs(self.path_to_results)
                os.makedirs(self.path_to_classifier_outputs)
                os.makedirs(self.path_to_datamodel_outputs)
                os.makedirs(self.path_to_clustering_outputs)
                os.makedirs(self.path_to_benchmarks)
                os.makedirs(self.path_to_plots)
                os.makedirs(self.path_to_snr_outputs)


@chz.chz
class Experiment(ExperimentBase):
    """
    Class containing all the necessary information for running an experiment.

    Attributes inherited from ExperimentBase.
    ----------------------------------------
    dataset : DatasetInterface
        The dataset used for the experiment.
    experiment_name : str
        Name of the experiment.
    path : Path
        Path to store experiment results if not in memory.
    notes : str
        Notes for the experiment.
    feature_selector : SelectPCsInterface
        Feature selector for the experiment.

    Attributes
    ----------
    datamodels_pipeline: DatamodelsPipelineInterface
        Datamodels pipeline for the experiment.
    tc_random_generator : Type[RandomGeneratorTC]
        Random generator for TC pipelines.
    mask_factory : subgroups.samplers.mask_factory_fn
        Factory for generating masks.
    model_factory : model_factory_fn
        Factory for creating models.
    sample_selector: IndicesFunction
        Returns indices to fit for the experiment.
    n_features : int
        Number of features to use for the experiment.
    """
    # Training Args
    mask_factory: mask_factory_fn = chz.field(default=mask_factory_fixed_alpha, doc='Factory for generating masks. This will be used for training the models.')
    model_factory: model_factory_fn = chz.field(default=model_factory_xgboost, doc='Factory for creating models. This will be used for training the models.')
    datamodel_factory: datamodel_factory_fn = chz.field(default=datamodel_factory_linear, doc='Factory for creating models. This will be used for training the models.')
    tc_random_generator: Type[RandomGeneratorTC]=chz.field(default=RandomGeneratorTC, doc='Random generator for TC pipelines. Will return independent random seeds for each component of the TC experiment, based on a batch starter seed.')
    n_features: int = chz.field(default=None, doc='Number of features to use for the experiment.')

    @chz.init_property
    def npcs(self):
        if self.n_features is None:
            return self.dataset.num_features
        else:
            return self.n_features

@chz.chz
class SNRExperiment(ExperimentBase):
    """
    Class containing all the necessary information for running an experiment.

    Attributes inherited from ExperimentBase.
    ----------------------------------------
    dataset : DatasetInterface
        The dataset used for the experiment.
    experiment_name : str
        Name of the experiment.
    path : Path
        Path to store experiment results if not in memory.
    notes : str
        Notes for the experiment.
    feature_selector : SelectPCsInterface
        Feature selector for the experiment.

    Attributes
    ----------
    model_factory_initializer : model_factory_init_fn
        Initializer for the model factory.
    mask_factory_initializer : mask_factory_init_fn
        Initializer for the mask factory.
    snr_n_models : int
        Number of models to build from ModelFactory. Each model will be trained on a different mask from MaskFactory.
    snr_n_passes : int
        Number of model initializations for SNR.
    npcs_min: int
        Minimum number of components required for each SNR.
    npcs_max: int
        Maximum number of components required for each SNR.
    snr_random_generator : Type[RandomGeneratorSNR]
        Random generator for SNR pipelines.
    stopping_condition : StoppingConditionInterface
        Stopping condition for the SNR experiment.

    """

    # SNR experiment Args
    model_factory_initializer: model_factory_init_fn=chz.field(default=None, doc='Initializer for the model factory. This will be used to sample instances of ModelFactory with different hyperparameters for the SNR experiment.')
    mask_factory_initializer: mask_factory_init_fn=chz.field(default=None, doc='Initializer for the mask factory. This will be used to sample instances of MaskFactory with different hyperparameters for the SNR experiment.')
    snr_n_models: int=chz.field(default=20, doc='Number of models to build from ModelFactory. Each model will be trained on a different mask from MaskFactory.')
    snr_n_passes: int=chz.field(default=15, doc='Number of passes over the same mask matrix for the SNR experiment (the number of times ModelFactory.build_model(seed=i) will be called with different seeds).')
    npcs_min: int=chz.field(default=5, doc='Minimum number of PCs to use for the SNR experiment.')
    npcs_max: int=chz.field(default=50, doc='Maximum number of PCs to use for the SNR experiment.')
    snr_random_generator: Type[RandomGeneratorSNR]=chz.field(default=None, doc='Random generator for SNR pipelines. Will return independent random seeds for each component of the SNR experiment, based on a batch starter seed.')
    stopping_condition: StoppingConditionInterface=chz.field(default=None, doc='Stopping condition for the SNR experiment.')

    