import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, List, Tuple 


def plot_reliability_diagram(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    bins: Optional[List[float]] = None,
    n_bins: int = 20,
    figsize: Tuple[int, int] = (15, 5)
) -> float:
    """
    Plot a reliability diagram with frequency plot
    
    Args:
        y_true: binary array (0 or 1)
        y_prob: array of probability of being class 1
        bins: custom bins
        n_bins: number of bins
        figsize: figure size
        
    Returns:
        mean square error between empirical probability and predicted probability
    """

    if bins is None:
        bins = np.linspace(0, 1, n_bins + 1)
    
    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    freq = bin_total[nonzero]
    
    mse = ((prob_true - prob_pred) ** 2).mean()
    
    # plot
    fig, ax = plt.subplots(figsize=figsize, ncols=2)
    _ = ax[0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    _ = ax[0].plot(prob_pred, prob_true, "s-", label="model")
    _ = ax[0].set_xlabel("Predicted")
    _ = ax[0].set_ylabel("Empirical")
    _ = ax[0].set_title("Reliability diagram")
    _ = ax[0].legend(loc="lower right")
    
    _ = ax[1].hist(y_prob)
    _ = ax[1].set_xlabel("Predicted")
    _ = ax[1].set_ylabel("Frequency")
    _ = ax[1].set_title("Frequency of the predicted probability")
    
    return mse
