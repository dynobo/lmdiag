import itertools

import numpy as np
import pytest
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

from lmdiag.lm_stats.linearmodels import LinearmodelsStats
from lmdiag.lm_stats.statsmodels import StatsmodelsStats


# TODO: Add test for 3D X
@pytest.mark.parametrize(
    "attr",
    [
        "residuals",
        "fitted_values",
        "standard_residuals",
        "cooks_d",
        "leverage",
        "params_count",
    ],
)
def test_lm_stats_modules(
    attr: str,
    statsmodels_lm: sm.OLS,
    linearmodels_lm: IV2SLS,
) -> None:
    model_stats_to_compare = [
        StatsmodelsStats(statsmodels_lm),
        LinearmodelsStats(linearmodels_lm),
    ]
    threshold = 1e-10

    for stats_a, stats_b in itertools.combinations(model_stats_to_compare, 2):
        distance = np.linalg.norm(getattr(stats_a, attr) - getattr(stats_b, attr))
        assert distance < threshold, (
            attr,
            stats_a.__class__.__name__,
            stats_b.__class__.__name__,
        )
