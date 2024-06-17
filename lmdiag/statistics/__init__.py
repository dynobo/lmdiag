import warnings
from typing import Any, Optional

import numpy as np
import statsmodels.api as sm

from lmdiag.statistics.base import StatsBase

try:
    import sklearn
except ImportError:
    sklearn = None

try:
    import linearmodels
except ImportError:
    linearmodels = None


def _warn_x_y_passed() -> None:
    warnings.warn(
        "`x` and `y` arguments are ignored for `statsmodels` and `linearmodels`; "
        "do not pass them.",
        stacklevel=3,
    )


def init_stats(
    lm: Any,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
) -> StatsBase:
    """Gather statistics depending on Linear Model type."""
    if isinstance(lm, sm.regression.linear_model.RegressionResultsWrapper):
        from lmdiag.statistics.statsmodels_stats import StatsmodelsStats

        if x or y:
            _warn_x_y_passed()

        return StatsmodelsStats(lm)

    if linearmodels is not None and isinstance(lm, linearmodels.iv.results.OLSResults):
        from lmdiag.statistics.linearmodels_stats import LinearmodelsStats

        if x or y:
            _warn_x_y_passed()

        return LinearmodelsStats(lm)

    if sklearn is not None and isinstance(lm, sklearn.linear_model.LinearRegression):
        from lmdiag.statistics.sklearn_stats import SklearnStats

        if x is None or y is None:
            raise ValueError("x and y args must be provided for sklearn models!")

        return SklearnStats(lm, x=x, y=y)

    raise TypeError(
        "Model type not (yet) supported. Currently supported are linear "
        "models from `statsmodels` and `linearmodels` packages."
    )
