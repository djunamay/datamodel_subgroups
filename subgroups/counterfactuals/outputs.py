
from subgroups.counterfactuals.base import ReturnCounterfactualOutputInterface
from subgroups.datastorage.mask_margin import MaskMarginStorageInterface
from typing import Callable
from subgroups.datastorage.mask_margin import MaskMarginStorageInterface
from subgroups.datastorage.experiment import Experiment
from subgroups.splits.base import SplitFactoryInterface
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from subgroups.datastorage.counterfactuals import CounterfactualOutputs
import numpy as np
from numpy.typing import NDArray
from typing import Optional

class ReturnCounterfactualOutputsBasic(ReturnCounterfactualOutputInterface):
    """
    Compute per-model counterfactual accuracies on complementary subsets of a given class.

    Given model outputs and a boolean split vector indicating 
    which samples of one class were available for training, this class computes model accuracy 
    on those held-out samples (split A) versus the complementary held-out samples of that class (split B).
    """
    def __init__(
        self,
        eligible_split_samples: Optional[NDArray[np.bool_]] = None,
    ):
        """
        Parameters
        ----------
        eligible_split_samples : NDArray[bool] or None
            If provided, defines which samples are eligible to belong to Split_B.
            Must be a boolean array of shape (n_samples,). If None, Split_B is
            defined as all samples of the same class that are not in Split_A.
        """
        self.eligible_split_samples = eligible_split_samples

    def __call__(self, training_output: MaskMarginStorageInterface, split: NDArray[bool]) -> CounterfactualOutputs:
        """
        Returns class CounterfactualOutputs based on input training data.

        Parameters
        ----------

        training_output : MaskMarginStorageInterface 
            Model training outputs for training regime where for split_class only samples in split were fair game (for the opposite class all samples were fair game)
                    
        split : NDArray of bool, shape (n_samples)
            Indicates which subset of split_class samples were fair game for training.
            
        Returns
        -------

        CounterfactualOutputs with attributes 
            acc_on_split_A : NDArray of float, shape (n_models,), accuracy computed on samples indexed by split vector AND NOT included in training.
            acc_on_split_B : NDArray of float, shape (n_models,), accuracy computed on samples of class split_class AND NOT indexed by split vector AND NOT included in training.
        """

        margins, masks = training_output.margins, training_output.masks
        split_class = np.unique(training_output.labels[split]) 

        if len(split_class)>1:
            raise ValueError('Split vector should index a subset of class 0 OR 1, right now it indexes both classes.')

        split_A, split_B = self._return_split_indices(split, training_output.labels, split_class)

        if not np.array_equal(
            np.unique(training_output.labels[split_A]),
            np.unique(training_output.labels[split_B])
        ):
            raise ValueError('Split_A and Split_B should index the same class.')

        if len(np.unique(training_output.masks[:,split_A]))==1:
            raise ValueError('Split_A was not used to train any classifiers.')

        if True in np.unique(training_output.masks[:,split_B]):
            raise ValueError('Split_B was used to train on. Only Split_A should be used for training.')

        logits_split_A, logits_split_B = self._return_logits(split_A, split_B, split_class, margins)
        masks_split_A, masks_split_B = masks[:,split_A], masks[:, split_B]
        labels_split_A, labels_split_B = np.repeat(split_class, split_A.sum()), np.repeat(split_class, split_B.sum())

        acc_on_split_A = self._return_accuracies(labels_split_A, logits_split_A, masks_split_A)
        acc_on_split_B = self._return_accuracies(labels_split_B, logits_split_B, masks_split_B)
        acc_diff = acc_on_split_A - acc_on_split_B

        return CounterfactualOutputs(acc_on_split_A = acc_on_split_A, acc_on_split_B = acc_on_split_B, score = acc_diff)
      
    
    def _return_split_indices(self, split, labels, split_class):
        """
        Returns boolean vectors corresponding to split A and split B samples. 

        Split A samples are indexed by the boolean split vector.
        Split B samples are indexed by ~split AND labels (if split_class is True), otherwise by  ~split AND ~labels (if split_class is False).

        Parameters
        ----------

        split_class : bool
                indicates which class the samples indexed by split vector belong to. 

        split : NDArray of bool, shape (n_samples)
            Indicates which subset of split_class samples were fair game for training.
        
        training_output : MaskMarginStorageInterface 
            Model training outputs for training regime where for split_class only samples in split were fair game (for the opposite class all samples were fair game)
        """
        index_split_A = split
        same_class_mask = labels == split_class

        if self.eligible_split_samples is None:
            index_split_B = ~split & same_class_mask
        else:
            index_split_B = ~split & same_class_mask & self.eligible_split_samples
        return index_split_A, index_split_B

    @staticmethod
    def _return_logits(split_A, split_B, split_class, margins):
        """
        Compute logits from margins for split A and split B samples.

        Parameters
        ----------
        margins : NDArray of float, shape (n_models, n_samples). 
            Model confidences for each sample.
        
        split_A : NDArray of bool, shape (n_samples_split_A, )

        split_B : NDArray of bool, shape (n_samples_split_B, )

        split_class : bool
            indicates which class samples of split_A and _B belong to. 

        """
        logits_split_A, logits_split_B = margins[:,split_A]/(2*split_class-1), margins[:,split_B]/(2*split_class-1)
        return logits_split_A, logits_split_B

    @staticmethod
    def _return_accuracies(labels, logits, masks):
        """
        Compute per-model classification accuracy.

        For each model (row), compares the predicted logits against the true labels
        on unmasked samples only. Masked entries indicate samples used for training 
        that should be excluded from evaluation.

        Parameters
        ----------
        masks : NDArray of bool, shape (n_models, n_samples)
            Masks indicatating samples excluded from evaluation for the corresponding model.
            
        logits : NDArray of float, shape (n_models, n_samples)
            Predicted logits (unnormalized scores) for each sample. 

        labels : NDArray of bool, shape (n_samples,)
            True labels for each sample.

        Returns
        -------
        accuracies : np.ndarray of float, shape (n_models,)
            Accuracy of each model, computed as the fraction of correctly classified 
            (unmasked) samples.
        """
        return np.array([accuracy_score(labels[np.invert(masks[i])], 
            logits[i][np.invert(masks[i])]>0) 
            for i in range(logits.shape[0])])