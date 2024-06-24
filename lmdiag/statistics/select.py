import warnings
from typing import Any, Optional

import numpy as np
from statsmodels.genmod.generalized_linear_model import GLMResults
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.robust.robust_linear_model import RLMResults

from lmdiag.statistics.base import StatsBase

try:
    import sklearn
except ImportError:
    sklearn = None

try:
    import linearmodels
except ImportError:
    linearmodels = None


def _warn_x_y() -> None:
    warnings.warn(
        "`x` and `y` arguments are ignored for this model type. Do not pass them.",
        stacklevel=3,
    )


def _init_linearmodels_stats(lm: Any) -> StatsBase:
    from lmdiag.statistics.linearmodels_stats import LinearmodelsStats

    return LinearmodelsStats(lm)


def _init_sklearn_stats(lm: Any, x: np.ndarray, y: np.ndarray) -> StatsBase:
    from lmdiag.statistics.sklearn_stats import SklearnStats

    return SklearnStats(lm, x=x, y=y)


def _init_statsmodels_stats(lm: Any) -> StatsBase:
    from lmdiag.statistics.statsmodels_stats import StatsmodelsStats

    return StatsmodelsStats(lm)


def get_stats(
    lm: Any, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None
) -> StatsBase:
    """Gather statistics depending on linear model type."""
    if isinstance(lm, (RegressionResultsWrapper, GLMResults, RLMResults)):
        if x or y:
            _warn_x_y()
        model_stats = _init_statsmodels_stats(lm)

    elif linearmodels and isinstance(
        lm, (linearmodels.iv.results.OLSResults, linearmodels.iv.results.IVResults)
    ):
        if x or y:
            _warn_x_y()
        model_stats = _init_linearmodels_stats(lm)

    elif sklearn and isinstance(lm, sklearn.linear_model.LinearRegression):
        if x is None or y is None:
            raise ValueError("x and y args must be provided this model type!")
        model_stats = _init_sklearn_stats(lm, x, y)

    else:
        raise TypeError(
            "Model type not (yet) supported. Currently supported are linear "
            "models from `statsmodels`, `linearmodels` and `sklearn` packages."
        )

    return model_stats
