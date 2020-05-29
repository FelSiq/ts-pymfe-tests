"""Utility functions used ubiquitously over this library source code."""
import typing as t
import operator

import sklearn.preprocessing
import numpy as np
import pandas as pd


def standardize_ts(ts: np.ndarray,
                   ts_scaled: t.Optional[np.ndarray] = None) -> np.ndarray:
    """Standardize (z-score normalization) time-series."""
    if ts_scaled is None:
        if not isinstance(ts, np.ndarray):
            ts = np.asarray(ts, dtype=float)

        if ts.ndim == 1:
            ts = ts.reshape(-1, 1)

        return sklearn.preprocessing.StandardScaler().fit_transform(ts).ravel()

    return ts


def find_plateau_pt(arr: np.ndarray,
                    arr_diff: t.Optional[np.ndarray] = None) -> np.ndarray:
    """Find plateau points in array.

    ``arr_diff`` is the first-order differenced ``arr``, which can be
    passed to speed up this computation since this value is needed within
    this function.
    """
    if arr_diff is None:
        arr_diff = np.diff(arr)

    arr_diff_2 = np.diff(arr_diff)

    res = np.logical_and(np.isclose(arr_diff_2, 0),
                         np.isclose(arr_diff[:-1], 0))

    return np.hstack((False, res, False))


def find_crit_pt(arr: np.ndarray, type_: str) -> np.ndarray:
    """Find critical points on the given values.

    ``type`` must be in {"min", "max", "plateau", "non-plateau", "any"}.
    """
    if arr.size <= 2:
        raise ValueError("Array too small (size {}). Need at least "
                         "3 elements.".format(arr.size))

    VALID_TYPES = {"min", "max", "plateau", "non-plateau", "any"}

    if type_ not in VALID_TYPES:
        raise ValueError("'type_' must be in {} (got '{}')."
                         "".format(type_, VALID_TYPES))

    # Note: first discrete derivative
    arr_diff_1 = np.diff(arr)

    if type_ == "plateau":
        return find_plateau_pt(arr, arr_diff_1)

    turning_pt = arr_diff_1[1:] * arr_diff_1[:-1] < 0

    if type_ == "non-plateau":
        return np.hstack((False, turning_pt, False))

    if type_ == "any":
        plat = find_plateau_pt(arr, arr_diff_1)
        turning_pt = np.hstack((False, turning_pt, False))
        res = np.logical_or(turning_pt, plat)
        return res

    # Note: second discrete derivative
    arr_diff_2 = np.diff(arr_diff_1)

    rel = operator.lt if type_ == "max" else operator.gt

    interest_pt = rel(arr_diff_2, 0)
    local_m = np.logical_and(turning_pt, interest_pt)

    return np.hstack((False, local_m, False))


def discretize(ts: np.ndarray,
               num_bins: int,
               strategy: str = "equal-width",
               dtype: t.Type = int) -> np.ndarray:
    """Discretize a time-series using a histogram.

    Parameters
    ----------
    ts : :obj:`np.ndarray`
        One-dimensional time-series values.

    num_bins : int
        Number of bins in the histogram.

    strategy : {`equal-width`,`equiprobable`}, optional (default="equal-width")
            Strategy used to define the histogram bins. Must be either
            `equal-width` (bins with equal with) or `equiprobable` (bins
            with the same amount of observations within).

    dtype : type, optional (default=int)
        Output type of the discretized time-series.

    Returns
    -------
    :obj:`np.ndarray`
        Discretized time-series with the selected strategy.
    """
    VALID_METHODS = {"equal-width", "equiprobable"}

    if strategy not in VALID_METHODS:
        raise ValueError("'strategy' must be in {} (got {})."
                         "".format(VALID_METHODS, strategy))

    if strategy == "equal-width":
        bins = np.histogram(ts, num_bins)[1][:-1]

    elif strategy == "equiprobable":
        bins = np.quantile(ts, np.linspace(0, 1, num_bins + 1)[:-1])

    ts_disc = np.digitize(ts, bins)

    return ts_disc.astype(dtype)
