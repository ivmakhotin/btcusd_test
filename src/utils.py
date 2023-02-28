from typing import Tuple

import numpy as np
import yaml


def window_sum(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    """

    c = np.nancumsum(x)
    s = c[w - 1:]
    s[1:] -= c[:-w]
    return s

def window_lin_reg(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    """

    # Invalidate both x and y values when there's a nan in one of them
    valid = np.isfinite(x) & np.isfinite(y)
    x[~valid] = np.nan
    y[~valid] = np.nan

    # Sums for each window
    n = window_sum(valid, w)  # Count only valid points in the window
    sx = window_sum(x, w)
    sy = window_sum(y, w)
    sx2 = window_sum(x ** 2, w)
    sxy = window_sum(x * y, w)

    # Avoid warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * sxy - sx * sy) / (n * sx2 - sx ** 2)
        intercept = (sy - slope * sx) / n

    # Replace infinities by nans. Not necessary, but cleaner.
    invalid_results = n < 2
    slope[invalid_results] = np.nan
    intercept[invalid_results] = np.nan

    return slope, intercept

def get_config():
    with open("src/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config