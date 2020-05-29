"""Module dedicated to autocorrelation time-series meta-features."""
import typing as t

import statsmodels.tsa.stattools
import numpy as np

import tspymfe._embed as _embed
import tspymfe._utils as _utils
import tspymfe._detrend as _detrend


class MFETSAutocorr:
    """Extract time-series meta-features from Autocorr group."""
    @classmethod
    def precompute_detrended_acf(cls,
                                 ts: np.ndarray,
                                 nlags: t.Optional[int] = None,
                                 unbiased: bool = True,
                                 **kwargs) -> t.Dict[str, np.ndarray]:
        """Precompute the detrended autocorrelation function.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        nlags : int, optional
            Number of lags to calculate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        kwargs:
            Additional arguments and previous precomputed items. May
            speed up this precomputation.

        Returns
        -------
        dict
            The following precomputed item is returned:
                * ``detrended_acfs`` (:obj:`np.ndarray`): the autocorrelation
                    function from the detrended time-series.
        """
        precomp_vals = {}

        if "detrended_acfs" not in kwargs:
            precomp_vals["detrended_acfs"] = cls.ft_acf_detrended(
                ts=ts, nlags=nlags, unbiased=unbiased)

        return precomp_vals

    @classmethod
    def _calc_acf(cls,
                  ts: np.ndarray,
                  nlags: t.Optional[int] = None,
                  unbiased: bool = True,
                  detrend: bool = True,
                  detrended_acfs: t.Optional[np.ndarray] = None,
                  ts_detrended: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Precompute the autocorrelation function.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        nlags : int, optional
            Number of lags to calculate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        detrend : bool, optional (default=True)
            If True, detrend the time-series using Friedman's Super Smoother
            before calculating the autocorrelation function, or the user
            given detrended time-series from ``ts_detrended`` argument.

        detrended_acfs : :obj:`np.ndarray`, optional
            This method's return value. Used to take advantage of
            precomputations.

        ts_detrended : :obj:`np.ndarray`, optional
            Detrended time-series. Used only if `detrend` is False.

        Returns
        -------
        :obj:`np.ndarray`
            If `detrend` is True, the autocorrelation function up to `nlags`
            lags of the detrended time-series. If `detrend` is False, the
            autocorrelation function up to `nlags` lags of the time-series.
        """
        if detrended_acfs is not None and (nlags is None
                                           or detrended_acfs.size == nlags):
            return detrended_acfs

        if detrend and ts_detrended is None:
            try:
                ts_detrended = _detrend.decompose(ts=ts, ts_period=0)[2]

            except ValueError:
                pass

        if ts_detrended is None:
            ts_detrended = ts

        if nlags is None:
            nlags = ts.size // 2

        acf = statsmodels.tsa.stattools.acf(ts_detrended,
                                            nlags=nlags,
                                            unbiased=unbiased,
                                            fft=True)
        return acf[1:]

    @classmethod
    def _first_acf_below_threshold(
        cls,
        ts: np.ndarray,
        threshold: float,
        abs_acf_vals: bool = False,
        max_nlags: t.Optional[int] = None,
        unbiased: bool = True,
        detrended_acfs: t.Optional[np.ndarray] = None,
    ) -> t.Union[int, float]:
        """First autocorrelation lag below a given threshold.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        threshold : float
            The threshold to find the first lag below it.

        abs_acf_vals : bool, optional (default=False)
            If True, avaliate the aboslute value of the autocorrelation
            function.

        max_nlags : int, optional
            Number of lags to avaluate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        detrended_acfs : :obj:`np.ndarray`, optional
            This method's return value. Used to take advantage of
            precomputations.

        Returns
        -------
        int or float
            Lag corresponding to the first autocorrelation function the
            given ``threshold``, if any. Return `np.nan` if no such index is
            found.
        """
        detrended_acfs = cls._calc_acf(ts=ts,
                                       nlags=max_nlags,
                                       unbiased=unbiased,
                                       detrended_acfs=detrended_acfs)

        if abs_acf_vals:
            # Note: in this case, we are testing if
            # -threshold <= acf <= threshold.
            detrended_acfs = np.abs(detrended_acfs)

        nonpos_acfs = np.flatnonzero(detrended_acfs <= threshold)

        try:
            return nonpos_acfs[0] + 1

        except IndexError:
            return np.nan

    @classmethod
    def ft_acf(cls,
               ts: np.ndarray,
               nlags: t.Optional[int] = None,
               unbiased: bool = True) -> np.ndarray:
        """Autocorrelation function of the time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        nlags : int, optional
            Number of lags to calculate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        Returns
        -------
        :obj:`np.ndarray`
            The autocorrelation function up to `nlags` lags of the time-series.
        """
        return cls._calc_acf(ts=ts,
                             nlags=nlags,
                             unbiased=unbiased,
                             detrend=False)

    @classmethod
    def ft_acf_detrended(
            cls,
            ts: np.ndarray,
            nlags: t.Optional[int] = None,
            unbiased: bool = True,
            ts_detrended: t.Optional[np.ndarray] = None,
            detrended_acfs: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Autocorrelation function of the detrended time-series.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        nlags : int, optional
            Number of lags to calculate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        ts_detrended : :obj:`np.ndarray`, optional
            Detrended time-series. If not given, the time-series is detrended
            within this method using Friedman's Super Smoother.

        detrended_acfs : :obj:`np.ndarray`, optional
            This method's return value. Used to take advantage of
            precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            The autocorrelation function up to `nlags` lags of the detrended
            time-series.
        """
        return cls._calc_acf(ts=ts,
                             nlags=nlags,
                             unbiased=unbiased,
                             detrend=True,
                             detrended_acfs=detrended_acfs,
                             ts_detrended=ts_detrended)

    @classmethod
    def ft_acf_first_nonsig(
        cls,
        ts: np.ndarray,
        max_nlags: t.Optional[int] = None,
        unbiased: bool = True,
        threshold: t.Optional[t.Union[int, float]] = None,
        detrended_acfs: t.Optional[np.ndarray] = None,
    ) -> t.Union[int, float]:
        """First non-significative detrended autocorrelation lag.

        The critical value to determine if a autocorrelation is significative
        is 1.96 / sqrt(len(ts)), but can be changed using the ``threshold``
        parameter.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        max_nlags : int, optional
            Number of lags to avaluate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        threshold : int or float, default
            The critical value to determine if a autocorrelation value is
            significative or not. This means that any autocorrelation with
            absolute value higher than is considered significative. If None,
            then the threshold used will be 1.96 / sqrt(len(ts)).

        ts_detrended : :obj:`np.ndarray`, optional
            Detrended time-series. Used only if `detrend` is False. If not
            given, the time-series is detrended within this method using
            Friedman's Super Smoother.

        Returns
        -------
        int or float
            Lag corresponding to the first autocorrelation with absolute value
            below the given ``threshold``, if any. Return `np.nan` if no such
            index is found.
        """
        if threshold is None:
            threshold = 1.96 / np.sqrt(ts.size)

        res = cls._first_acf_below_threshold(ts=ts,
                                             threshold=threshold,
                                             abs_acf_vals=True,
                                             max_nlags=max_nlags,
                                             unbiased=unbiased,
                                             detrended_acfs=detrended_acfs)
        return res

    @classmethod
    def ft_acf_first_nonpos(
        cls,
        ts: np.ndarray,
        max_nlags: t.Optional[int] = None,
        unbiased: bool = True,
        detrended_acfs: t.Optional[np.ndarray] = None,
    ) -> t.Union[int, float]:
        """First non-positive detrended autocorrelation lag.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        max_nlags : int, optional
            Number of lags to avaluate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        detrended_acfs : :obj:`np.ndarray`, optional
            Detrended time-series autocorrelation function with each index
            corresponding to its lag starting from the lag 1.

        Returns
        -------
        int or float
            Lag corresponding to the first autocorrelation below or equal
            zero, if any. Return `np.nan` if no such index is found.
        """
        res = cls._first_acf_below_threshold(ts=ts,
                                             threshold=0,
                                             abs_acf_vals=False,
                                             max_nlags=max_nlags,
                                             unbiased=unbiased,
                                             detrended_acfs=detrended_acfs)
        return res

    @classmethod
    def ft_first_acf_locmin(
        cls,
        ts: np.ndarray,
        max_nlags: t.Optional[int] = None,
        unbiased: bool = True,
        detrended_acfs: t.Optional[np.ndarray] = None,
    ) -> t.Union[int, float]:
        """First local minima detrended autocorrelation lag.

        Parameters
        ----------
        ts : :obj:`np.ndarray`
            One-dimensional time-series values.

        max_nlags : int, optional
            Number of lags to avaluate the autocorrelation function.

        unbiased : bool, optional (default=True)
            If True, the autocorrelation function is corrected for statistical
            bias.

        detrended_acfs : :obj:`np.ndarray`, optional
            Detrended time-series autocorrelation function with each index
            corresponding to its lag starting from the lag 1.

        Returns
        -------
        int or float
            Lag corresponding to the first autocorrelation below or equal
            zero, if any. Return `np.nan` if no such index is found.
        """
        detrended_acfs = cls._calc_acf(ts=ts,
                                       nlags=max_nlags,
                                       unbiased=unbiased,
                                       detrended_acfs=detrended_acfs)

        acfs_locmin = np.flatnonzero(
            _utils.find_crit_pt(detrended_acfs, type_="min"))

        try:
            return acfs_locmin[0] + 1

        except IndexError:
            return np.nan
