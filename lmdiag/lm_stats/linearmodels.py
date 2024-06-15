import linearmodels
import numpy as np

from lmdiag.lm_stats.base import StatsBase, optionally_cached_property


class LinearmodelsStats(StatsBase):
    def __init__(
        self,
        lm: linearmodels.iv.results.OLSResults,
        cache: bool = True,
    ) -> None:
        super().__init__()
        self._lm = lm
        self._cache_properties = cache

    @optionally_cached_property
    def residuals(self) -> np.ndarray:
        return self._lm.resids.values

    @optionally_cached_property
    def fitted_values(self) -> np.ndarray:
        return self._lm.fitted_values.squeeze().values

    @optionally_cached_property
    def standard_residuals(self) -> np.ndarray:
        residuals = self.residuals
        h_ii = self.leverage
        var_e = np.sqrt(self._lm.resid_ss / (self._lm.nobs - self._lm.df_model))
        standard_error = var_e * np.sqrt(1 - h_ii)
        return residuals / standard_error

    @optionally_cached_property
    def cooks_d(self) -> np.ndarray:
        h_ii = self.leverage
        cooks_d2 = self.standard_residuals**2 / self.params_count
        cooks_d2 *= h_ii / (1 - h_ii)
        return cooks_d2

    @optionally_cached_property
    def leverage(self) -> np.ndarray:
        X = self._lm.model._x
        XtX_inv = np.linalg.inv(np.dot(X.T, X))
        return np.einsum("ij,jk,ik->i", X, XtX_inv, X)

    @optionally_cached_property
    def params_count(self) -> int:
        return len(self._lm.params)
