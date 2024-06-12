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
        vals = OLSInfluence(self._lm).summary_frame()
        return vals["standard_resid"].values

    @optionally_cached_property
    def cooks_d(self) -> np.typing.ArrayLike:
        vals = OLSInfluence(self._lm).summary_frame()
        return vals["cooks_d"].values

    @optionally_cached_property
    def leverage(self) -> np.typing.ArrayLike:
        influence = self._lm.get_influence()
        return influence.hat_matrix_diag

    @optionally_cached_property
    def params_count(self) -> int:
        return len(self._lm.params)
