import chz
from typing import Protocol, Any, Optional
from ..datasets import DatasetInterface
from ..datasamplers import MaskFactory, MaskFactoryInitializer
from ..models import ModelFactory, ModelFactoryInitializer
from pathlib import Path
import os
from ..datasamplers.random_generators import RandomGeneratorSNRInterface, RandomGeneratorTCInterface
from ..experiments.stopping_condition import StoppingConditionInterface
from typing import Type
from ..datamodels.base import DatamodelsPipelineInterface
from ..datamodels.indices import IndicesFunction
from ..datasamplers.feature_selectors import SelectPCsInterface
#from ..counterfactuals.base import CounterfactualInputsInterface, CounterfactualEvaluationInterface

@chz.chz
class Experiment:
    """
    Class containing all the necessary information for running an experiment.

    Attributes
    ----------
    dataset : DatasetInterface
        The dataset used for the experiment.
    mask_factory : MaskFactory
        Factory for generating masks.
    model_factory : ModelFactory
        Factory for creating models.
    model_factory_initializer : ModelFactoryInitializer
        Initializer for the model factory.
    mask_factory_initializer : MaskFactoryInitializer
        Initializer for the mask factory.
    snr_n_models : int
        Number of models to build from ModelFactory. Each model will be trained on a different mask from MaskFactory.
    snr_n_passes : int
        Number of model initializations for SNR.
    snr_random_generator : Type[RandomGeneratorSNRInterface]
        Random generator for SNR experiments.
    tc_random_generator : Type[RandomGeneratorTCInterface]
        Random generator for TC experiments.
    in_memory : bool
        Flag indicating if the experiment should be conducted in memory.
    path : Path
        Path to store experiment results if not in memory.
    experiment_name : str
        Name of the experiment.
    stopping_condition : StoppingConditionInterface
        Stopping condition for the SNR experiment.
    """
    dataset: DatasetInterface=chz.field(default=None, doc='The dataset used for the experiment.')
    mask_factory: MaskFactory=chz.field(default=None, doc='Factory for generating masks. This will be used for training the classifier.')
    model_factory: ModelFactory=chz.field(default=None, doc='Factory for creating models. This will be used for training the classifier.')
    model_factory_initializer: ModelFactoryInitializer=chz.field(default=None, doc='Initializer for the model factory. This will be used to sample instances of ModelFactory with different hyperparameters for the SNR experiment.')
    mask_factory_initializer: MaskFactoryInitializer=chz.field(default=None, doc='Initializer for the mask factory. This will be used to sample instances of MaskFactory with different hyperparameters for the SNR experiment.')
    snr_n_models: int=chz.field(default=20, doc='Number of models to build from ModelFactory. Each model will be trained on a different mask from MaskFactory.')
    snr_n_passes: int=chz.field(default=15, doc='Number of passes over the same mask matrix for the SNR experiment (the number of times ModelFactory.build_model(seed=i) will be called with different seeds).')
    in_memory: bool=chz.field(default=True, doc='Flag indicating if the results to any experiment pipelines run on this experiment object will be stored in memory.')
    path: str=chz.field(default=None, doc='Path to store experiment results if not in memory.')
    experiment_name: str=chz.field(default=None, doc='Name of the experiment if not in memory.')
    snr_random_generator: Type[RandomGeneratorSNRInterface]=chz.field(default=None, doc='Random generator for SNR experiments. Will return independent random seeds for each component of the SNR experiment, based on a batch starter seed.')
    tc_random_generator: Type[RandomGeneratorTCInterface]=chz.field(default=None, doc='Random generator for TC experiments. Will return independent random seeds for each component of the TC experiment, based on a batch starter seed.')
    stopping_condition: StoppingConditionInterface=chz.field(default=None, doc='Stopping condition for the SNR experiment.')
    datamodels_pipeline: DatamodelsPipelineInterface=chz.field(default=None, doc='Datamodels pipeline for the experiment.')
    dm_n_train: int=chz.field(default=None, doc='Number of training samples for the datamodels pipeline.')
    dm_n_test: Optional[int]=chz.field(default=None, doc='Number of test samples for the datamodels pipeline.')
    indices_to_fit: IndicesFunction=chz.field(default=None, doc='Indices to fit for the experiment.')
    notes: str=chz.field(default=None, doc='Notes for the experiment.')
    npcs_min: int=chz.field(default=5, doc='Minimum number of PCs to use for the SNR experiment.')
    npcs_max: int=chz.field(default=50, doc='Maximum number of PCs to use for the SNR experiment.')
    feature_selector: SelectPCsInterface=chz.field(default=None, doc='Feature selector for the experiment.')
    npcs: int=chz.field(default=None, doc='Number of PCs to use for the experiment.')
   # counterfactual_inputs: None #CounterfactualInputsInterface=chz.field(default=None, doc='Counterfactual inputs for the experiment.')
   # counterfactual_estimator: None #CounterfactualEvaluationInterface=chz.field(default=None, doc='Counterfactual estimator for the experiment.')

    @chz.init_property
    def _check_path(self):
        """
        Check if the path to the experiment results is provided if in_memory is False.
        """
        if not self.in_memory and self.path is None:
            raise ValueError("path to output must be provided if in_memory is False")
        if self.path is not None and self.experiment_name is None:
            raise ValueError("experiment_name must be provided if path is provided")

    @chz.init_property
    def path_to_results(self):
        """
        Path to the results of the experiment.
        """
        if self.in_memory:
            return None
        else:
            return os.path.join(self.path, self.experiment_name)

    @chz.init_property
    def path_to_classifier_outputs(self):
        """
        Path to the classifier outputs of the experiment.
        """
        if self.in_memory:
            return None
        else:
            return os.path.join(self.path_to_results, 'classifier_outputs')

    @chz.init_property
    def path_to_datamodel_outputs(self):
        """
        Path to the datamodel outputs of the experiment.
        """
        if self.in_memory:
            return None
        else:
            return os.path.join(self.path_to_results, 'datamodel_outputs')
    
    @chz.init_property
    def path_to_clustering_outputs(self):
        """
        Path to the clustering outputs of the experiment.
        """
        if self.in_memory:
            return None
        else:
            return os.path.join(self.path_to_results, 'clustering_outputs')
    
    @chz.init_property
    def path_to_benchmarks(self):
        """
        Path to the benchmarks of the experiment.
        """
        if self.in_memory:
            return None
        else:
            return os.path.join(self.path_to_results, 'benchmarks')

    @chz.init_property
    def path_to_plots(self):
        """
        Path to the plots of the experiment.
        """
        if self.in_memory:
            return None
        else:
            return os.path.join(self.path_to_results, 'plots')

    @chz.init_property
    def path_to_snr_outputs(self):
        """
        Path to the SNR outputs of the experiment.
        """
        if self.in_memory:
            return None
        else:
            return os.path.join(self.path_to_results, 'snr_outputs')

    @chz.init_property
    def _prepare_experiment(self):
        """
        If not in memory, prepare an experiment with the given configuration.
        """
        if not self.in_memory:
            if not os.path.exists(self.path_to_results):
                os.makedirs(self.path_to_results)
                os.makedirs(self.path_to_classifier_outputs)
                os.makedirs(self.path_to_datamodel_outputs)
                os.makedirs(self.path_to_clustering_outputs)
                os.makedirs(self.path_to_benchmarks)
                os.makedirs(self.path_to_plots)
                os.makedirs(self.path_to_snr_outputs)

    
    