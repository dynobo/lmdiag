import itertools
from typing import Callable

import numpy as np
import pytest

from lmdiag.lm_stats.linearmodels import LinearmodelsStats
from lmdiag.lm_stats.statsmodels import StatsmodelsStats


@pytest.mark.parametrize("x_dims", [1])
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
    statsmodels_factory: Callable,
    linearmodels_factory: Callable,
    x_dims: int,
) -> None:
    model_stats_to_compare = [
        StatsmodelsStats(statsmodels_factory(x_dims=x_dims)),
        LinearmodelsStats(linearmodels_factory(x_dims=x_dims)),
    ]

    acceptable_distance = 1e-8

    for stats_a, stats_b in itertools.combinations(model_stats_to_compare, 2):
        distance = np.linalg.norm(getattr(stats_a, attr) - getattr(stats_b, attr))
        assert distance < acceptable_distance, (
            attr,
            stats_a.__class__.__name__,
            stats_b.__class__.__name__,
        )
