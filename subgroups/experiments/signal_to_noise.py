import chz
from tqdm import tqdm
import numpy as np
from typing import Callable, Type
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from subgroups.datasets import DatasetInterface
from subgroups.dataloaders import DataloaderInterface
from subgroups.models.classifier import ClassifierInterface

@chz.chz
class SignalToNoiseInterface(ABC):
    """
    Interface for per-sample *signal* and *noise* estimates from an ensemble of trained models.

    Definitions
    -----------

    Let:
        x : a single held-out sample
        Y : scalar predicted margin for sample x
        T : a random subset of training samples
        θ : random model initialization (seed) for a given training subset T

    Signal: S(x) = Var_T( E_θ[ Y | x, T ] )
        Variance of the mean predicted margin (averaged over seeds θ) for sample x across models trained on different training subsets T.
        Captures variance in predictions due to changes in training data.

    Noise: N(x) = E_T( Var_θ[ Y | x, T ] )
        Expected variance of predicted margins for sample x over different model seeds θ, given a fixed training subset T, across models trained on different training subsets T.
        Captures variance in predictions unexplained by training-set differences (i.e., model randomness).

    Shapes
    ------
    Evaluating signal and noise across multiple training subsets or architectures yields arrays of shape:
        (n_models, n_samples)

    Implementations must expose the two properties below.
    """

    @property
    @abstractmethod
    def signal(self) -> NDArray[float]:
        """Per-model, per-sample signal estimates (shape: n_models × n_samples)."""
        ...

    @property
    @abstractmethod
    def noise(self) -> NDArray[float]:
        """Per-model, per-sample noise estimates (shape: n_models × n_samples)."""
        ...

@chz.chz
class SignalToNoiseExperiment(SignalToNoiseInterface):
    dataset: DatasetInterface=chz.field
    classifier: Type[ClassifierInterface]=chz.field
    dataloader: Type[DataloaderInterface]=chz.field
    n_model_inits: int=chz.field
    n_train_splits: int=chz.field
    alpha: float=chz.field
    show_progress: bool=chz.field(default=True)
    classifier_kwargs: dict=chz.field(default_factory=dict)

    @chz.init_property
    def _mask(self)-> NDArray[bool]:
        mask = np.zeros((self.n_train_splits, self.n_model_inits, self.dataset.features.shape[0]), dtype=bool)
        return mask
    
    @chz.init_property
    def _margins(self)-> NDArray[bool]:
        margins = np.empty((self.n_train_splits, self.n_model_inits, self.dataset.features.shape[0]), dtype=float)
        return margins
    
    @chz.init_property
    def _masked_margins(self)-> NDArray[float]:
        margins = self._margins
        mask = self._mask 
        rng_train = np.arange(self.n_train_splits)
        rng_model = np.arange(self.n_model_inits)

        iterator = (
            tqdm(rng_train, desc="training splits", disable=not self.show_progress)
            if self.show_progress else rng_train
        )

        for i in iterator:                        # loop over resampled training sets
            dl = self.dataloader(dataset=self.dataset, 
                            alpha=self.alpha, 
                            train_seed=i)
            for j in rng_model:                   # loop over model inits for that split
                clf = self.classifier(
                    dataloader=dl,
                    model_seed=j,
                    **self.classifier_kwargs,
                )
                margins[i, j] = clf.margins
                mask[i, j, dl.train_indices] = True

        return np.ma.array(margins, mask=mask)
        
    @chz.init_property
    def signal(self)-> NDArray[float]:
        expectation = self._masked_margins.mean(axis=1)
        signal = np.array(np.var(expectation, axis=0))
        return signal
    
    @chz.init_property
    def noise(self)-> NDArray[float]:
        variance = self._masked_margins.var(axis=1)
        noise = np.array(np.mean(variance, axis=0))
        return noise
    
@chz.chz
class SignalToNoiseArgs:
    dataset_factory: Callable[[], DatasetInterface] = chz.field(
        doc="Callable that returns a dataset instance"
    )
    classifier: Type[ClassifierInterface]=chz.field
    classifier_kwargs: Callable[[], dict]=chz.field
    dataloader: Type[DataloaderInterface]=chz.field
    n_model_inits: int=chz.field
    n_train_splits: int=chz.field
    alpha: float=chz.field
    show_progress: bool=chz.field(default=True)

def run_SNR(args: SignalToNoiseArgs):
    exp = SignalToNoiseExperiment(dataset=args.dataset_factory(), classifier=args.classifier, dataloader=args.dataloader, 
                                  n_model_inits=args.n_model_inits, n_train_splits=args.n_train_splits, alpha=args.alpha, 
                                  show_progress=args.show_progress, classifier_kwargs=args.classifier_kwargs())
    return exp.signal/exp.noise
    
if __name__ == "__main__":
    chz.nested_entrypoint(run_SNR)


