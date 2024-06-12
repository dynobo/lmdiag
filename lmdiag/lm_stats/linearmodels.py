from typing import TYPE_CHECKING

import linearmodels
import numpy as np
import pandas as pd

from lmdiag.lm_stats.base import StatsBase, optionally_cached_property

if TYPE_CHECKING:
    import linearmodels


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
        return self._lm.resids

    @optionally_cached_property
    def fitted_values(self) -> np.ndarray:
        fitted = self._lm.fitted_values

        # Transform series to 1-d array, if necessary
        if isinstance(fitted, pd.core.frame.DataFrame):
            fitted = fitted.values[:, 0]

        return fitted

    @optionally_cached_property
    def standard_residuals(self) -> np.ndarray:
        x = self._lm.model._x[:, 1]
        mean_x = np.mean(x)
        diff_mean_sqr = np.dot((x - mean_x), (x - mean_x))
        residuals = self.residuals
        h_ii = (x - mean_x) ** 2 / diff_mean_sqr + (1 / self._lm.nobs)
        var_e = np.sqrt(self._lm.resid_ss / (self._lm.nobs - 2))
        se_regression = var_e * ((1 - h_ii) ** 0.5)
        return residuals / se_regression

    @optionally_cached_property
    def cooks_d(self) -> np.ndarray:
        x = self._lm.model._x[:, 1]
        mean_x = np.mean(x)
        diff_mean_sqr = np.dot((x - mean_x), (x - mean_x))
        h_ii = (x - mean_x) ** 2 / diff_mean_sqr + (1 / self._lm.nobs)
        cooks_d2 = self.standard_residuals**2 / self.params_count
        cooks_d2 *= h_ii / (1 - h_ii)
        return cooks_d2

    @optionally_cached_property
    def leverage(self) -> np.ndarray:
        x = self._lm.model._x[:, 1]
        mean_x = np.mean(x)
        diff_mean_sqr = np.dot((x - mean_x), (x - mean_x))
        h_ii = (x - mean_x) ** 2 / diff_mean_sqr + (1 / self._lm.nobs)
        return h_ii

    @optionally_cached_property
    def params_count(self) -> int:
        return len(self._lm.params)
