"""Time-series embedding functions."""
import typing as t

import numpy as np
import scipy.spatial

try:
    import tspymfe._utils as _utils

except ImportError:
    pass

try:
    import tspymfe.autocorr as autocorr

except ImportError:
    pass


def embed_ts(ts: np.ndarray,
             dim: int,
             lag: int = 1,
             include_val: bool = True) -> np.ndarray:
    """Embbed a time-series in dimension ``dim``.

    Arguments
    ---------
    ts : :obj:`np.ndarray`, shape: (ts.size,)
        One-dimensional time-series.

    dim : int
        Dimension of the embedding.

    lag : int, optional (default=1)
        Lag of the time-series.

    include_val : bool, optional (default=False)
        Include the value itself on its own embedding.

    Returns
    -------
    :obj:`np.ndarray`, shape: (ts.size - dim * lag, dim)
        Embbed time-series.
    """
    if dim <= 0:
        raise ValueError("'dim' must be positive (got {}).".format(dim))

    if lag <= 0:
        raise ValueError("'lag' must be positive (got {}).".format(lag))

    if dim * lag > ts.size:
        raise ValueError("'dim * lag' ({}) can't be larger than the "
                         "time-series length ({}).".format(dim * lag, ts.size))

    if include_val:
        dim -= 1

    ts_emb = np.zeros((ts.size - dim * lag, dim + int(include_val)),
                      dtype=ts.dtype)

    shift_inds = lag * (dim - 1 - np.arange(-int(include_val), dim))

    for i in np.arange(ts_emb.shape[0]):
        ts_emb[i, :] = ts[i + shift_inds]

    return ts_emb


def nn(embed: np.ndarray,
       metric: str = "chebyshev",
       p: t.Union[int, float] = 2) -> np.ndarray:
    """Return the Nearest neighbor of each embedded time-series observation."""
    dist_mat = scipy.spatial.distance.cdist(embed, embed, metric=metric, p=p)

    # Note: prevent nearest neighbor be the instance itself, and also
    # be exact equal instances. We follow Cao's recommendation to pick
    # the next nearest neighbor when this happens.
    dist_mat[np.isclose(dist_mat, 0.0)] = np.inf

    nn_inds = np.argmin(dist_mat, axis=1)

    return nn_inds, dist_mat[nn_inds, np.arange(nn_inds.size)]


def embed_dim_cao(
    ts: np.ndarray,
    lag: int,
    dims: t.Union[int, t.Sequence[int]] = 16,
    ts_scaled: t.Optional[np.ndarray] = None,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Estimate Cao's metrics to estimate time-series embedding dimension.

    The Cao's metrics are two statistics, `E1` and `E2`, used to estimate the
    appropriate embedding metric of a time-series. From the `E1` statistic it
    can be defined the appropriate embedding dimension as the index after the
    saturation of the metric from a set of ordered lags.

    The precise `saturation` concept may be a subjective concept, since this
    metric can show some curious `artifacts` related to specific lags for
    specific time-series, which will need deeper further investigation.

    The `E2` statistics is to detect `false positives` from the `E1` statistic
    since if is used to distinguish between random white noise and a process
    generated from a true, non completely random, underlying process. If the
    time-series is purely random white noise, then all values of `E2` will be
    close to 1. If there exists a dimension with the `E2` metric estimated
    `sufficiently far` from 1, then this series is considered not a white
    random noise.

    Parameters
    ----------
    ts : :obj:`np.ndarray`
        One-dimensional time-series values.

    lag : int
        Embedding lag. You may want to check the `embed_lag` function
        documentation for embedding lag estimation. Must be a stricly
        positive value.

    dims : int or sequence of int
        Dimensions to estimate the Cao's `E1` and `E2` statistic values.
        If integer, estimate all dimensions from 1 up to the given number.
        If a sequence of integers, estimate all Cao's statistics for all
        given dimensions, and return the corresponding values in the same
        order of the given dimensions.
        All dimensions with non-positive values will receive a `np.nan`
        value for both Cao's metric.

    ts_scaled : :obj:`np.ndarray`, optional
        Standardized time-series values. Used to take advantage of
        precomputations.

    Returns
    -------
    tuple of :obj:`np.ndarray`
        `E1` and `E2` Cao's metrics, necessarily in that order, for all
        given dimensions (and with direct index correspondence for the
        given dimensions).

    References
    ----------
    .. [1] Liangyue Cao, Practical method for determining the minimum
        embedding dimension of a scalar time series, Physica D: Nonlinear
        Phenomena, Volume 110, Issues 1–2, 1997, Pages 43-50,
        ISSN 0167-2789, https://doi.org/10.1016/S0167-2789(97)00118-8.
    """
    if lag <= 0:
        raise ValueError("'lag' must be positive (got {}).".format(lag))

    _dims: t.Sequence[int]

    if np.isscalar(dims):
        _dims = np.arange(1, int(dims) + 1)  # type: ignore

    else:
        _dims = np.asarray(dims, dtype=int)

    ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

    ed, ed_star = np.zeros((2, len(_dims)), dtype=float)

    for ind, dim in enumerate(_dims):
        try:
            emb_next = embed_ts(ts=ts_scaled, lag=lag, dim=dim + 1)
            emb_cur = emb_next[:, 1:]

        except ValueError:
            ed[ind] = np.nan
            ed_star[ind] = np.nan
            continue

        nn_inds, dist_cur = nn(embed=emb_cur)

        emb_next_abs_diff = np.abs(emb_next[:, 0] - emb_next[nn_inds, 0])
        # Note: 'chebyshev'/'manhattan'/'L1'/max norm distance of X and Y,
        # both in the embed of (d + 1) dimensions, can be defined in respect
        # to one dimension less:
        # L1(X_{d+1}, Y_{d+1}) = |X_{d+1}, Y_{d+1}|_{inf}
        #   = max(|x_1 - y_1|, ..., |x_{d+1} - y_{d+1}|)
        #   = max(max(|x_1 - y_1|, ..., |x_d - y_d|), |x_{d+1} - y_{d+1}|)
        #   = max(L1(X_{d}, Y_{d}), |x_{d+1} - y_{d+1}|)
        dist_next = np.maximum(dist_cur, emb_next_abs_diff)

        # Note: 'ed' and 'ed_star' refers to, respectively, E_{d} and
        # E^{*}_{d} from the Cao's paper.
        ed[ind] = np.mean(dist_next / dist_cur)
        ed_star[ind] = np.mean(emb_next_abs_diff)

    # Note: the minimum embedding dimension is D such that e1[D]
    # is the first index where e1 stops changing significantly.
    e1 = ed[1:] / ed[:-1]

    # Note: This is the E2(d) Cao's metric. Its purpose is to
    # separate random time-series. For random-generated time-
    # series, e2 will be 1 for any dimension. For deterministic
    # data, however, e2 != 1 for some d.
    e2 = ed_star[1:] / ed_star[:-1]

    return e1, e2


def embed_lag(ts: np.ndarray,
              lag: t.Optional[t.Union[str, int]] = None,
              default_lag: int = 1,
              max_nlags: t.Optional[int] = None,
              detrended_acfs: t.Optional[np.ndarray] = None,
              detrended_ami: t.Optional[np.ndarray] = None,
              **kwargs) -> int:
    """Find the appropriate embedding lag using a given criteria.

    Parameters
    ----------
    ts : :obj:`np.ndarray`
        One-dimensional time-series values.

    lag : int or str, optional (default = None)
        If scalar, return its own value casted to integer,

        If string, it must be one value in {`ami`, `acf`, `acf-nonsig`},
        which defines the strategy of defining the appropriate lag of
        the embedding.
            1. `ami`: uses the first minimum lag of the automutual information
                of the time-series.
            2. `acf`: uses the first negative lag of the autocorrelation of the
                time-series.
            3. `acf-nonsig` (default): uses the first non-significant lag of
                the time-series autocorrelation function. The non-significant
                value is defined as the first lag that has the absolute value
                of is autocorrelation below the critical value defined as
                1.96 / sqrt(ts.size).

        If None, the lag will be searched will the 'acf-nonsig'
        criteria.

    max_nlags : int, optional
        If ``lag`` is not a numeric value, than it will be estimated using
        either the time-series autocorrelation or mutual information
        function estimated up to this argument value.

    detrended_acfs : :obj:`np.ndarray`, optional
        Array of time-series autocorrelation function (for distinct ordered
        lags) of the detrended time-series. Used only if ``lag`` is any of
        `acf`, `acf-nonsig` or None.  If this argument is not given and the
        previous condiditon is meet, the autocorrelation function will be
        calculated inside this method up to ``max_nlags``.

    detrended_ami : :obj:`np.ndarray`, optional
        Array of time-series automutual information function (for distinct
        ordered lags). Used only if ``lag`` is `ami`. If not given and the
        previous condiditon is meet, the automutual information function
        will be calculated inside this method up to ``max_nlags``.

    kwargs:
        Extra arguments for the function used to estimate the lag. used
        only if `lag` is not a numeric value.

    Returns
    -------
    int
        Estimated embedding lag.

    Notes
    -----
    This method may be used to estimate `auto-interations` of the time-series
    (such as calculating the autocorrelation function, for instance) aswell.
    """
    VALID_OPTIONS = {
        "acf": autocorr.MFETSAutocorr.ft_acf_first_nonpos,
        "acf-nonsig": autocorr.MFETSAutocorr.ft_acf_first_nonsig,
    }  # type: t.Dict[str, t.Callable[..., t.Union[float, int]]]

    if lag is None:
        lag = "acf-nonsig"

    if isinstance(lag, str):
        if lag not in VALID_OPTIONS:
            raise ValueError("'lag' must be in {} (got '{}')."
                             "".format(VALID_OPTIONS.keys(), lag))

        if max_nlags is None:
            max_nlags = ts.size // 2

        if lag == "ami":
            kwargs["detrended_ami"] = detrended_ami

        else:
            kwargs["detrended_acfs"] = detrended_acfs

        kwargs["max_nlags"] = max_nlags

        estimated_lag = VALID_OPTIONS[lag](ts, **kwargs)

        return default_lag if np.isnan(estimated_lag) else int(estimated_lag)

    if np.isscalar(lag):
        lag = int(lag)

        if lag <= 0:
            raise ValueError("'lag' must be positive (got {}).".format(lag))

        return lag

    raise TypeError("'lag' type must be a scalar, a string or None (got {})."
                    "".format(type(lag)))


def ft_emb_dim_cao(ts: np.ndarray,
                   dims: t.Union[int, t.Sequence[int]] = 16,
                   lag: t.Optional[t.Union[str, int]] = None,
                   tol_threshold: float = 0.05,
                   check_e2: bool = True,
                   max_nlags: t.Optional[int] = None,
                   ts_scaled: t.Optional[np.ndarray] = None,
                   detrended_acfs: t.Optional[np.ndarray] = None,
                   detrended_ami: t.Optional[np.ndarray] = None,
                   emb_dim_cao_e1: t.Optional[np.ndarray] = None,
                   emb_dim_cao_e2: t.Optional[np.ndarray] = None) -> int:
    """Embedding dimension estimation using Cao's method.

    Using the Cao's embedding dimension estimation, it is calculated both
    of its metrics, `E1` and `E2` whose purpose is to, respectively,
    detect the appropriate embedding dimension and detect whether the given
    time-series is generated by a completely random process (white noise).

    The appropriate embedding dimension is the saturation dimension of `E1`
    if and only if exists a dimension `E2` sufficiently distinct from 1.
    If `E2` is approximately constant at 1 over all dimensions, the series
    is considered white noise and, therefore, the embedding dimension is
    assumed to be 1.

    Parameters
    ----------
    ts : :obj:`np.ndarray`
        One-dimensional time-series values.

    dims : int or a sequence of int, optional (default=16)
        The embedding dimension candidates. In int, investigate all values
        between 1 and ``dims`` value (both inclusive). If a sequence of
        integers is used, then investigate only the given dimensions.

    lag : int or str, optional
        Lag of the time-series embedding. It must be a strictly positive
        value, None or a string in {`acf`, `acf-nonsig`, `ami`}. In the
        last two type of options, the lag is estimated within this method
        using the given strategy method (or, if None, it is used the
        strategy `acf-nonsig` by default) up to ``max_nlags``.
            1. `acf`: the lag corresponds to the first non-positive value
                in the autocorrelation function.
            2. `acf-nonsig`: lag corresponds to the first non-significant
                value in the autocorrelation function (absolute value below
                the critical value of 1.96 / sqrt(ts.size)).
            3. `ami`: lag corresponds to the first local minimum of the
                time-series automutual information function.

    tol_threshold : float, optional (default=0.05)
        Tolerance threshold to defined the maximum absolute diference
        between two E1 values in order to assume saturation. This same
        threshold is the minimum absolute deviation that E2 values must
        have in order to be considered different than 1.

    check_e2 : bool, optional (default=True)
        If True, check if there exist a Cao's E2 value different than 1,
        and return 1 if this condition is not satisfied. If False, ignore
        the E2 values.

    max_nlags : int, optional
        If ``lag`` is not a numeric value, than it will be estimated using
        either the time-series autocorrelation or mutual information
        function estimated up to this argument value.

    ts_scaled : :obj:`np.ndarray`, optional
        Standardized time-series values. Used to take advantage of
        precomputations.

    detrended_acfs : :obj:`np.ndarray`, optional
        Array of time-series autocorrelation function (for distinct ordered
        lags) of the detrended time-series. Used only if ``lag`` is any of
        `acf`, `acf-nonsig` or None.  If this argument is not given and the
        previous condiditon is meet, the autocorrelation function will be
        calculated inside this method up to ``max_nlags``.

    detrended_ami : :obj:`np.ndarray`, optional
        Array of time-series automutual information function (for distinct
        ordered lags). Used only if ``lag`` is `ami`. If not given and the
        previous condiditon is meet, the automutual information function
        will be calculated inside this method up to ``max_nlags``.

    emb_dim_cao_e1 : :obj:`np.ndarray`, optional
        E1 values from the Cao's method. Used to take advantage of
        precomputations.

    emb_dim_cao_e2 : :obj:`np.ndarray`, optional
        E2 values from the Cao's method. Used to take advantage of
        precomputations.

    Returns
    -------
    int
        Estimation of the appropriate embedding dimension using Cao's
        method.

    References
    ----------
    .. [1] Liangyue Cao, Practical method for determining the minimum
        embedding dimension of a scalar time series, Physica D: Nonlinear
        Phenomena, Volume 110, Issues 1–2, 1997, Pages 43-50,
        ISSN 0167-2789, https://doi.org/10.1016/S0167-2789(97)00118-8.
    """
    ts_scaled = _utils.standardize_ts(ts=ts, ts_scaled=ts_scaled)

    lag = embed_lag(ts=ts_scaled,
                    lag=lag,
                    detrended_acfs=detrended_acfs,
                    detrended_ami=detrended_ami,
                    max_nlags=max_nlags)

    if emb_dim_cao_e1 is None or (check_e2 and emb_dim_cao_e2 is None):
        emb_dim_cao_e1, emb_dim_cao_e2 = embed_dim_cao(ts=ts,
                                                       ts_scaled=ts_scaled,
                                                       dims=dims,
                                                       lag=lag)

    if (check_e2 and emb_dim_cao_e2 is not None
            and np.all(np.abs(emb_dim_cao_e2 - 1) < tol_threshold)):
        return 1

    e1_abs_diff = np.abs(np.diff(emb_dim_cao_e1))

    first_max_ind = 0

    try:
        first_max_ind = np.flatnonzero(e1_abs_diff <= tol_threshold)[0]

    except IndexError:
        pass

    return first_max_ind + 1
