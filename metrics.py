import numpy as np

def interval_score(x, lower, upper, alpha=0.05):
    """
    Computes the interval score for predictive intervals.

    Args:
        x (numpy.ndarray): Ground truth values.
        lower (numpy.ndarray): Lower bound of the predictive interval.
        upper (numpy.ndarray): Upper bound of the predictive interval.
        alpha (float): Significance level (default: 0.05).

    Returns:
        numpy.ndarray: Interval scores for each prediction.
    """
    # Ensure that the upper bound is greater than or equal to the lower bound
    assert np.all(upper >= lower), "Upper bound must be >= lower bound."
    
    # Calculate interval score
    return (
        (upper - lower) +  # Width of the interval
        (2 / alpha) * (lower - x) * (x < lower) +  # Penalty for underestimation
        (2 / alpha) * (x - upper) * (x > upper)    # Penalty for overestimation
    )
