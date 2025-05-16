from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
from ..utils.scoring import compute_signal_noise

class StoppingConditionInterface(ABC):
    """
    Stopping condition interface. A class that implements this interface takes as input the masks and margins returned by the function compute_snr_for_one_architecture and returns a boolean value indicating whether the stopping condition is met.
    """

    @abstractmethod
    def evaluate_stopping(self, margins, masks) -> bool:
        """
        Evaluate the stopping condition.

        Parameters
        ----------
        margins : np.ndarray 
            The margins matrix output by the function compute_snr_for_one_architecture.
            The shape of the matrix is (n_models, n_samples, n_passes).
        masks : np.ndarray
            The masks matrix output by the function compute_snr_for_one_architecture.
            The shape of the matrix is (n_models, n_samples, n_passes).

        Returns
        -------
        bool
            True if the stopping condition is met, False otherwise.
        """
        ...

class SNRPrecisionStopping(StoppingConditionInterface):
    """
    Stop when the mean-SNR estimate is precise to ± `tolerance` %.

    Procedure
    ---------
    1. **Point estimate**
       Compute a single plug-in estimate

           snr_hat = np.mean(compute_signal_noise(margins, masks))

       where *margins* has shape (n_passes, n_models, n_samples).

    2. **Bootstrap sampling**
       Draw *B* bootstrap replicates by resampling from n_models
       with replacement; for each replicate recompute the population-level SNR as above.
       The 2.5 % and 97.5 % empirical quantiles give a 95 %
       confidence interval, `[lower, upper]`.

    3. **Stopping rule**
       Let the relative half-width be

           hw_rel = (upper - lower) / (2 * abs(snr_hat))

       Stop (return `True`) when

           hw_rel <= tolerance/100            # e.g. 0.05 for ±5 %

    Notes
    -----
    * The denominator is the plug-in estimate `snr_hat`, not the
      mean of the bootstrap replicates.
    * The rule guarantees that, with ≈95 % coverage, the true SNR
      lies within ± `tolerance` % of `snr_hat`.

    Attributes
    ----------
    B : int
        Number of bootstrap samples.
    tolerance : float
        Tolerance for the stopping condition.
    """

    def __init__(self, B=1000, tolerance=0.01):
        self.B = B
        self.tolerance = tolerance

    @staticmethod
    def _E_init_V_init(margins, masks):
        """
        Computes expectation and variance across model initializations.

        Parameters
        ----------
        margins : np.ndarray
            The margins matrix output by the function compute_snr_for_one_architecture.
            The shape of the matrix is (n_passes, n_models, n_samples).
        masks : np.ndarray
            The masks matrix output by the function compute_snr_for_one_architecture.
            The shape of the matrix is (n_passes, n_models, n_samples).

        Returns
        -------
        E_init : np.ndarray
            The expectation across model initializations.
            The shape is (n_models, n_samples)
        V_init : np.ndarray
            The variance across model initializations.
            The shape is (n_models, n_samples)
        """
        margins_masked = np.ma.array(margins, mask=masks)
        E_init = np.ma.mean(margins_masked, axis=0)
        V_init = np.ma.var(margins_masked, axis=0, ddof=1)
        return E_init, V_init
    
    @staticmethod
    def _compute_snr(E_init, V_init, axis=0):
        """
        Returns the signal-to-noise ratio along the first axis of the input arrays.

        Parameters
        ----------
        E_init : np.ndarray
            The expectation across model initializations.
            The shape is (n_models, n_samples)
        V_init : np.ndarray
            The variance across model initializations.
            The shape is (n_models, n_samples)

        Returns
        -------
        signal/noise : np.ndarray
            The signal-to-noise ratio along the first axis of the input arrays.
            The shape is (n_samples,)
        """
        signal = np.ma.var(E_init, axis=axis, ddof=1)
        noise = np.ma.mean(V_init, axis=axis)
        return signal/np.ma.maximum(noise, 1e-12)
    
    @staticmethod
    def _get_bootstrap_index(E_init, B):
        """
        Draws B bootstrap sample indices, returning an array of shape (B, n_models)
        """
        N = E_init.shape[0]
        return np.random.randint(0, N, (B, N))

    @staticmethod
    def _confidence_interval(bootstrap_storage, lower_bound=2.5, upper_bound=97.5):
        """
        Compute the 95% confidence interval of the signal-to-noise ratio on the bootstrap samples.
        Input array has shape (B,)
        """
        valid = bootstrap_storage.compressed() 
        lower, upper = np.percentile(valid, lower_bound), np.percentile(valid, upper_bound)
        return lower, upper 
    
    @staticmethod
    def _relative_precision(lower, upper, estimate, eps=1e-12):
        return (upper - lower) / (2 * max(estimate, eps))
    
    def evaluate_stopping(self, margins, masks):
        """
        Evaluate the stopping condition.
        
        Parameters
        ----------
        margins : np.ndarray
            The margins matrix output by the function compute_snr_for_one_architecture.
            The shape of the matrix is (n_passes, n_models, n_samples).
        masks : np.ndarray
            The masks matrix output by the function compute_snr_for_one_architecture.
            The shape of the matrix is (n_passes, n_models, n_samples).

        Returns
        -------
        bool
            True if the stopping condition is met, False otherwise.
        """
        E_init, V_init = self._E_init_V_init(margins, masks) 
        index = self._get_bootstrap_index(E_init, self.B)
        bootstrap_snr = self._compute_snr(E_init[index], V_init[index], axis=1)
        population_bootstrap_snr = bootstrap_snr.mean(axis=1)
        lower, upper = self._confidence_interval(population_bootstrap_snr)
        snr_hat = np.mean(self._compute_snr(E_init, V_init))
        relative_precision = self._relative_precision(lower, upper, snr_hat)

        print(f"The 95% CI of the SNR estimate is {np.round(snr_hat, 3)} ± {np.round(relative_precision*100, 3)} %")

        return relative_precision<self.tolerance
    
# class SNRPrecisionStopping(StoppingConditionInterface):
#     """
#     Stop when the mean-SNR estimate is precise to ± `tolerance` %.

#     Procedure
#     ---------
#     1. **Point estimate**
#        Compute a single plug-in estimate

#            snr_hat = np.mean(compute_signal_noise(margins, masks))

#        where *margins* has shape (n_models, n_samples, n_passes) and each row `i`
#        represents one training-subset used in a given model.

#     2. **Bootstrap sampling**
#        Draw *B* bootstrap replicates by resampling the **Y rows**
#        with replacement; for each replicate recompute the population-level SNR as above.
#        The 2.5 % and 97.5 % empirical quantiles give a 95 %
#        confidence interval, `[lower, upper]`.

#     3. **Stopping rule**
#        Let the relative half-width be

#            hw_rel = (upper - lower) / (2 * abs(snr_hat))

#        Stop (return `True`) when

#            hw_rel <= tolerance/100            # e.g. 0.05 for ±5 %

#     Notes
#     -----
#     * The denominator is the plug-in estimate `snr_hat`, not the
#       mean of the bootstrap replicates.
#     * The rule guarantees that, with ≈95 % coverage, the true SNR
#       lies within ± `tolerance` % of `snr_hat`.

#     Attributes
#     ----------
#     B : int
#         Number of bootstrap samples.
#     tolerance : float
#         Tolerance for the stopping condition.
#     """
#     def __init__(self, B=1000, tolerance=0.01):
#         self.B = B
#         self.tolerance = tolerance

#     @staticmethod
#     def _snr_estimate(margins, masks):
#         """
#         Compute the point estimate of the signal-to-noise ratio on the full margins and masks.
#         """
#         return compute_signal_noise(margins, masks)
    
#     @staticmethod
#     def _single_bootstrap(margins, masks):
#         """
#         Performs a single boostrap resampling of the margins and masks by sampling with replacement N indices from the range [0, N) (N = margins.shape[0]).
#         The signal-to-noise ratio is computed on the resampled margins and masks using the function compute_signal_noise.

#         Parameters
#         ----------
#         margins : np.ndarray 
#             The margins matrix output by the function compute_snr_for_one_architecture.
#             The shape of the matrix is (n_models, n_samples, n_passes).
#         masks : np.ndarray
#             The masks matrix output by the function compute_snr_for_one_architecture.
#             The shape of the matrix is (n_models, n_samples, n_passes).

#         Returns
#         -------
#         snr_hat : float
#             The signal-to-noise ratio of the resampled margins and masks averaged over n_samples.
#         """
#         N = margins.shape[0]
#         index = np.random.randint(0, N, N)
#         snr_hat = compute_signal_noise(margins[index], masks[index])
#         return np.mean(snr_hat)
    
#     @staticmethod
#     def _confidence_interval(bootstrap_storage, lower_bound=2.5, upper_bound=97.5):
#         """
#         Compute the 95% confidence interval of the signal-to-noise ratio on the bootstrap samples.
#         """
#         lower, upper = np.percentile(bootstrap_storage, lower_bound), np.percentile(bootstrap_storage, upper_bound)
#         return lower, upper 
    
#     @staticmethod
#     def _relative_precision(lower, upper, estimate, eps=1e-12):
#         return (upper - lower) / (2 * max(estimate, eps))
    
#     @staticmethod
#     def _decision_boundary(relative_precision, tolerance):
#         """
#         Check if the relative precision of the signal-to-noise ratio on the bootstrap samples is less than the tolerance.
#         """
#         return relative_precision<tolerance
    
#     def evaluate_stopping(self, margins, masks):
#         """
#         Evaluate the stopping condition.
#         """
#         snr_estimate = np.mean(self._snr_estimate(margins, masks))
#         bootstrap_storage = np.zeros((self.B))

#         for i in tqdm(range(self.B)):
#             bootstrap_storage[i] = self._single_bootstrap(margins, masks)
        
#         lower_ci, upper_ci = self._confidence_interval(bootstrap_storage)
#         print(snr_estimate)
#         relative_precision = self._relative_precision(lower_ci, upper_ci, snr_estimate)
#         return self._decision_boundary(relative_precision, self.tolerance), relative_precision*100
    