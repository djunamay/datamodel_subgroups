import numpy as np
import torchvision
from torchvision.transforms import ToTensor
from typing import List
from numpy.typing import NDArray
import chz
from .base import BaseDataset

@chz.chz
class CIFAR10Dataset(BaseDataset):
    """
    Processes the CIFAR10 dataset, providing access to features, labels, and metadata.

    Attributes
    ----------
    path_to_data : str
        Path to the CSF data.
    path_to_meta_data : str
        Path to the meta data.
    """
    path_to_cifar10: str = chz.field(doc="Path to the CIFAR10 data")
    download: bool = chz.field(doc="Download the CIFAR10 data", default=False)
    train: bool = chz.field(doc="Train or test set", default=False)

    @staticmethod
    def _to_numpy_image(tensor_image):
        numpy_image = ((tensor_image).numpy() * 255).astype(np.uint8)
        return numpy_image
    
    def _transform_one_image(self, image):
        return self._to_numpy_image(ToTensor()(image))

    @staticmethod
    def _get_binary_labels(targets, animate_labels: set = {2, 3, 4, 5, 6, 7}) -> List[bool]:
        binary_targets = [label in animate_labels for label in targets]
        return np.array(binary_targets)
    
    @chz.init_property
    def _dataset(self) -> torchvision.datasets.CIFAR10:
        return torchvision.datasets.CIFAR10(self.path_to_cifar10, train=self.train, download=self.download)
    
    @chz.init_property
    def _data(self):
        data = self._dataset.data.copy().astype(np.uint8).reshape(-1, 3, 32, 32)
        for i in range(len(self._dataset.data)):
            data[i] = self._transform_one_image(self._dataset.data[i])
        return data
    
    @chz.init_property
    def _targets(self) -> NDArray[int]:
        return np.array(self._dataset.targets)
    
    @chz.init_property
    def _binary_labels(self) -> List[bool]:
        return self._get_binary_labels(self._targets)
    
    @chz.init_property
    def _descriptive_data(self) -> np.recarray:
        target_to_name = {i: name for i, name in enumerate(self._dataset.classes)}
        labels = [target_to_name[x] for x in self._targets]
        labels_arr = np.array(labels, dtype='<U10')  
        return np.rec.fromarrays([labels_arr], names='class')
    
    @property
    def features(self) -> NDArray[float]:
        """
        Feature matrix (shape: [n_samples, n_features]).
        """
        return self._data

    @property
    def coarse_labels(self) -> NDArray[bool]:
        """
        Binary labels indicating dementia status (shape: [n_samples]).
        """
        return self._binary_labels

    @property
    def fine_labels(self) -> NDArray[bool]:
        """
        Integer labels for each unique primary CSF diagnostic label (shape: [n_samples]).
        """
        return self._targets

    @property
    def descriptive_data(self) -> np.recarray:
        """
        Descriptive data (shape: [n_samples, n_descriptive_features]).
        """
        return self._descriptive_data