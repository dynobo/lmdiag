import numpy as np
from sklearn.linear_model import LinearRegression

from lmdiag.statistics.base import StatsBase, optionally_cached_property


class SklearnStats(StatsBase):
    def __init__(
        self,
        lm: LinearRegression,
        x: np.ndarray,
        y: np.ndarray,
        cache: bool = True,
    ) -> None:
        super().__init__()
        self._lm = lm
        self._X = x
        self._y = y
        self._cache_properties = cache

    @optionally_cached_property
    def residuals(self) -> np.ndarray:
        return self._y - self.fitted_values

    @optionally_cached_property
    def fitted_values(self) -> np.ndarray:
        return self._lm.predict(self._X)

    @optionally_cached_property
    def standard_residuals(self) -> np.ndarray:
        residuals = self.residuals
        h_ii = self.leverage
        resid_ss: np.ndarray = np.sum(residuals**2)
        nobs = len(self._y)
        df_model = self.parameter_count
        var_e = np.sqrt(resid_ss / (nobs - df_model))
        standard_error = var_e * np.sqrt(1 - h_ii)
        return residuals / standard_error

    @optionally_cached_property
    def cooks_d(self) -> np.ndarray:
        h_ii = self.leverage
        cooks_d2 = self.standard_residuals**2 / self.parameter_count
        cooks_d2 *= h_ii / (1 - h_ii)
        return cooks_d2

    @optionally_cached_property
    def leverage(self) -> np.ndarray:
        X = self._X
        # add constant, like sm.add_constant() does in linearmodels' leverage()
        X = np.column_stack((np.ones(X.shape[0]), X))
        XtX_inv = np.linalg.inv(np.dot(X.T, X))
        h_ii = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        return h_ii

    @optionally_cached_property
    def parameter_count(self) -> int:
        params_count = len(self._lm.coef_)
        if self._lm.fit_intercept:
            params_count += 1
        return params_count
