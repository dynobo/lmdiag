from typing import Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence, summary_table

from lmdiag.lm_stats.base import StatsBase, optionally_cached_property


class StatsmodelsStats(StatsBase):
    def __init__(
        self,
        lm: sm.regression.linear_model.RegressionResultsWrapper,
        cache: bool = True,
    ) -> None:
        super().__init__()
        self._lm = lm
        self._cache_properties = cache
        self.__ols_influence: Union[OLSInfluence, None] = None

    @property
    def _ols_influence(self) -> OLSInfluence:
        if self.__ols_influence is None:
            self.__ols_influence = OLSInfluence(self._lm)
        return self.__ols_influence

    @optionally_cached_property
    def residuals(self) -> np.ndarray:
        _, data, _ = summary_table(self._lm, alpha=0.05)
        return data[:, 8]

    @optionally_cached_property
    def fitted_values(self) -> pd.Series:
        """Return 1-D numpy array with fitted values."""
        fitted = self._lm.fittedvalues
        # Transform series to 1-d array, if necessary
        if isinstance(fitted, pd.Series):
            fitted = fitted.values
        return fitted

    @optionally_cached_property
    def standard_residuals(self) -> np.typing.ArrayLike:
        return self._ols_influence.resid_studentized_internal

    @optionally_cached_property
    def cooks_d(self) -> np.typing.ArrayLike:
        return self._ols_influence.cooks_distance[0]

    @optionally_cached_property
    def leverage(self) -> np.typing.ArrayLike:
        return self._ols_influence.hat_matrix_diag

    @optionally_cached_property
    def params_count(self) -> int:
        return len(self._lm.params)
