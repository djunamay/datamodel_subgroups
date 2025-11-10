
from .base import ReturnCounterfactualOutputInterface
from ..datastorage.base import MaskMarginStorageInterface
from typing import Callable
from subgroups.datastorage.base import MaskMarginStorageInterface
from subgroups.datastorage.experiment import Experiment
from subgroups.counterfactuals.base import SplitFactoryInterface
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from ..datastorage.counterfactuals import CounterfactualOutputs
import numpy as np


class ReturnCounterfactualOutputsBasic(ReturnCounterfactualOutputInterface):
    def __call__(self, training_output: MaskMarginStorageInterface, n_models: int, split: NDArray[bool]) -> CounterfactualOutputs:

        masked_margins, masked_labels = np.ma.array(training_output.margins, mask=training_output.masks), np.ma.array(np.tile(training_output.labels.reshape(-1,1), n_models).T, mask=training_output.masks)
        labels_in_split, labels_out_split, labels_other_class, = masked_labels[:,split], masked_labels[:,~split & training_output.labels], masked_labels[:,~split & ~training_output.labels]
        logits_in_split, logits_out_split, logits_other_class = masked_margins[:,split]/(2*labels_in_split-1), masked_margins[:,~split & training_output.labels]/(2*labels_out_split-1), masked_margins[:,~split & ~training_output.labels]/(2*labels_other_class-1)

        return CounterfactualOutputs(acc_in = self._return_accuracies(labels_in_split, logits_in_split), acc_out = self._return_accuracies(labels_out_split, logits_out_split), acc_diff = self._return_accuracies(labels_in_split, logits_in_split) - self._return_accuracies(labels_out_split, logits_out_split))
      
    @staticmethod
    def _return_accuracies(labels, logits):
        return np.array([accuracy_score(labels[i].compressed(), 
            logits[i].compressed()>0) 
            for i in range(logits.shape[0])])