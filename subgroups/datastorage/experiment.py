import chz
from typing import Protocol, Any, Optional
from ..datasets import DatasetInterface
from ..datasamplers import MaskFactory
from ..models import ModelFactory, ModelFactoryInitializer
from pathlib import Path
import os
@chz.chz
class Experiment:
    dataset: DatasetInterface=chz.field(default=None)
    mask_factory: MaskFactory=chz.field(default=None)
    model_factory: ModelFactory=chz.field(default=None)
    model_factory_initializer: ModelFactoryInitializer=chz.field(default=None)
    in_memory: bool=chz.field(default=True)
    snr_n_train_splits: int=chz.field(default=20)
    snr_n_model_inits: int=chz.field(default=15)
    path: Path=chz.field(default=None)
    experiment_name: str=chz.field(default=None)

    @chz.init_property
    def _check_path(self):
        if not self.in_memory and self.path is None:
            raise ValueError("path to output must be provided if in_memory is False")
        if self.path is not None and self.experiment_name is None:
            raise ValueError("experiment_name must be provided if path is provided")

    @chz.init_property
    def path_to_results(self):
        if self.in_memory:
            return None
        else:
            return os.path.join(self.path, self.experiment_name)

    @chz.init_property
    def path_to_classifier_outputs(self):
        if self.in_memory:
            return None
        else:
            return os.path.join(self.path_to_results, 'classifier_outputs')

    @chz.init_property
    def path_to_datamodel_outputs(self):
        if self.in_memory:
            return None
        else:
            return os.path.join(self.path_to_results, 'datamodel_outputs')
    
    @chz.init_property
    def path_to_clustering_outputs(self):
        if self.in_memory:
            return None
        else:
            return os.path.join(self.path_to_results, 'clustering_outputs')
    
    @chz.init_property
    def path_to_benchmarks(self):
        if self.in_memory:
            return None
        else:
            return os.path.join(self.path_to_results, 'benchmarks')

    @chz.init_property
    def path_to_plots(self):
        if self.in_memory:
            return None
        else:
            return os.path.join(self.path_to_results, 'plots')

    @chz.init_property
    def path_to_snr_outputs(self):
        if self.in_memory:
            return None
        else:
            return os.path.join(self.path_to_results, 'snr_outputs')

    @chz.init_property
    def _prepare_experiment(self):
        """
        Prepare an experiment with the given configuration.
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

