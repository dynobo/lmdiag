import math
from typing import TYPE_CHECKING

import linearmodels
import numpy as np
import pandas as pd

from lmdiag.lm_stats.base import StatsBase

if TYPE_CHECKING:
    import linearmodels


class LinearmodelsStats(StatsBase):
    def __init__(self, lm: linearmodels.iv.results.OLSResults) -> None:
        super().__init__()
        self._lm = lm

    @property
    def residuals(self) -> np.ndarray:
        return self._lm.resids

    @property
    def fitted_values(self) -> np.ndarray:
        fitted = self._lm.fitted_values

        # Transform series to 1-d array, if necessary
        if isinstance(fitted, pd.core.frame.DataFrame):
            fitted = fitted.values[:, 0]

        return fitted

    @property
    def standard_residuals(self) -> np.ndarray:
        x = self._lm.model._x[:, 1]
        mean_x = np.mean(x)
        diff_mean_sqr = np.dot((x - mean_x), (x - mean_x))
        residuals = self.residuals
        h_ii = (x - mean_x) ** 2 / diff_mean_sqr + (1 / self._lm.nobs)
        # TODO: sqrt with numpy
        var_e = math.sqrt(self._lm.resid_ss / (self._lm.nobs - 2))
        se_regression = var_e * ((1 - h_ii) ** 0.5)
        return residuals / se_regression

    @property
    def cooks_d(self) -> np.ndarray:
        x = self._lm.model._x[:, 1]
        mean_x = np.mean(x)
        diff_mean_sqr = np.dot((x - mean_x), (x - mean_x))
        h_ii = (x - mean_x) ** 2 / diff_mean_sqr + (1 / self._lm.nobs)
        cooks_d2 = self.standard_residuals**2 / self.params_count
        cooks_d2 *= h_ii / (1 - h_ii)
        return cooks_d2

    @property
    def leverage(self) -> np.ndarray:
        x = self._lm.model._x[:, 1]
        mean_x = np.mean(x)
        diff_mean_sqr = np.dot((x - mean_x), (x - mean_x))
        h_ii = (x - mean_x) ** 2 / diff_mean_sqr + (1 / self._lm.nobs)
        return h_ii

    @property
    def params_count(self) -> int:
        # TODO: Check if this work
        return len(self._lm.params)
