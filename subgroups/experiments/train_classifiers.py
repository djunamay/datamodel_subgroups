from ..datasets import DatasetInterface
from ..datasamplers import MaskFactory
from ..models import ModelFactory
from ..utils.scoring import compute_margins
from ..datastorage.base import MaskMarginStorage
from typing import Optional
from pathlib import Path
import chz
import numpy as np
from ..models.base import SklearnClassifier
from numpy.typing import NDArray
from sklearn.utils import shuffle

@chz.chz
class TrainClassifiersArgs:
    dataset: DatasetInterface
    mask_factory: MaskFactory
    model_factory: ModelFactory
    n_models: int
    in_memory: bool = True
    path: Optional[Path] = None

def train_one_classifier(features: NDArray[np.float32], labels: NDArray[bool], mask: NDArray[bool], model: SklearnClassifier, seed: int):
    """
    Train a single classifier and return the margins and test accuracy.
    """
    features_shuffled, labels_shuffled = shuffle(features[mask], labels[mask], random_state=seed)
    model.fit(features_shuffled, labels_shuffled) 
    test_accuracy = model.score(features[~mask], labels[~mask])
    margins = compute_margins(model.predict_proba(features)[:,1], labels)
    return margins, test_accuracy

def train_classifiers(args: TrainClassifiersArgs):
    """
    Train multiple classifiers and store the results in a mask margin storage.
    """
    ds = args.dataset
    model_factory = args.model_factory
    n_models = args.n_models
    mask_margin_storage = MaskMarginStorage(n_models, ds.num_samples, ds.coarse_labels, args.mask_factory, args.in_memory, args.path)
    rng = args.mask_factory._rng

    for i in range(n_models):
        if mask_margin_storage.is_filled(i):
            continue
        else:
            model = model_factory.build_model() #TODO: this could mean that a model instance is repeated
            margins, test_accuracy = train_one_classifier(ds.features, ds.coarse_labels, mask_margin_storage.masks[i], model, rng.integers(0, 2**32 - 1))
            mask_margin_storage.fill_results(i, margins, test_accuracy)
    
    if mask_margin_storage.in_memory:
        return mask_margin_storage
    else:
        return mask_margin_storage.path

if __name__ == "__main__":
    chz.nested_entrypoint(train_classifiers)



