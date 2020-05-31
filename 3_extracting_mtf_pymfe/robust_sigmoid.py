"""Zero-centered Sigmoid function robust to outliers."""
import typing as t

import numpy as np
import pandas as pd


class RobustSigmoid:
    """Zero-centered Sigmoid function robust to outliers."""
    def __init__(self):
        self.iqr = np.empty(0, dtype=float)
        self.std = np.empty(0, dtype=float)
        self.mean = np.empty(0, dtype=float)
        self.median = np.empty(0, dtype=float)
        self.centers = np.empty(0, dtype=float)

        self.ids_robust = np.empty(0, dtype=bool)
        self.ids_trad = np.empty(0, dtype=bool)

    def fit(self, X: np.ndarray, _: t.Optional[np.ndarray] = None) -> np.ndarray:
        """Fit parameters using train data."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Note: '1.35' is the IQR of a Normal(0, 1) distribution.
        self.iqr = 1.35 * np.subtract(*np.quantile(X, (0.75, 0.25), axis=0))

        self.ids_trad = np.isclose(self.iqr, 0.0)
        self.ids_robust = ~self.ids_trad

        self.iqr = self.iqr[self.ids_robust]
        self.median = np.median(X[:, self.ids_robust], axis=0)

        self.mean = np.median(X[:, self.ids_trad], axis=0)
        self.std = np.std(X[:, self.ids_trad], axis=0)

        self.std[np.isclose(self.std, 0)] = 1.0

        # Note: the sigmoid function is in range [0, 1] and we're craving to
        # apply PCA, we need to center the transfomed data to the origin.
        self.centers = self.transform(X, center=False).mean(axis=0)

        return self

    def _transform(self,
                   X: np.ndarray,
                   copy: bool = True,
                   center: bool = True) -> np.ndarray:
        """Transform data using robust sigmoid."""
        if copy:
            X = np.copy(X)

        _np_err = np.geterr()
        np.seterr(over="ignore")

        X[:, self.ids_robust] = 1. / (1. + np.exp(
            -(X[:, self.ids_robust] - self.median) / self.iqr))
        X[:, self.ids_trad] = 1. / (1. + np.exp(
            -(X[:, self.ids_trad] - self.mean) / self.std))

        X[np.isclose(np.inf, X)] = 1.
        X[np.isclose(-np.inf, X)] = 0.

        np.seterr(**_np_err)

        if center:
            return X - self.centers

        return X

    def transform(self,
                  X: np.ndarray,
                  center: bool = True) -> np.ndarray:
        """Transform data using robust sigmoid."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return self._transform(X=X.values, center=center)

        return self._transform(X=X, center=center)


def _test():
    np.random.seed(16)
    X = np.hstack((np.random.random((5, 3)), np.array([1, 1, 1, 1,
                                                       2]).reshape(-1, 1)))
    tr = RobustSigmoid().fit(X)
    print(tr.transform(X).ptp(axis=0))


if __name__ == "__main__":
    _test()
