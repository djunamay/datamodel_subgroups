import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
import chz
from subgroups.train_classifier.base import DataloaderInterface

@chz.chz
class ClassifierInterface(ABC):
    """Abstract classifier: declare fields and required methods."""
    @property
    @abstractmethod
    def predictions(self)-> None:
        """Inferred probabilities for the full dataset"""
        ...

    @property
    @abstractmethod
    def margins(self)-> float:
        """Margins on the full dataset (all samples, test and train)"""
        ...
    
    @property
    @abstractmethod
    def accuracy(self)-> float:
        """Accuracy on the test dataset"""
        ...
    
@chz.chz
class BaseClassifier(ClassifierInterface):
    """Base classifier: define common functions."""
    dataloader: DataloaderInterface=chz.field

    @property
    def margins(self)-> float:
        """Evaluate the classifier"""
        logit_class_1 = np.log(self.predictions/(1-self.predictions))
        logit_class_0 = 1-self.predictions
        logit_class_0 = np.log(logit_class_0/(1-logit_class_0))

        margins = np.zeros_like(self.predictions, dtype=float)
        index_class1 = self.dataloader.dataset.class_indices[0]
        index_class0 = self.dataloader.dataset.class_indices[1]

        margins[index_class1] = logit_class_1[index_class1]-logit_class_0[index_class1]
        margins[index_class0] = logit_class_0[index_class0]-logit_class_1[index_class0]
        
        return margins

    @property
    def accuracy(self)-> float:
        """Evaluate the classifier"""
        predictions = self.predictions[self.dataloader.test_indices]>0.5
        true_labels = self.dataloader.dataset.coarse_labels[self.dataloader.test_indices]
        return np.mean(predictions == true_labels)