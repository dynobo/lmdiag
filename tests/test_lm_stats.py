import itertools
from typing import Callable

import numpy as np
import pytest

from lmdiag.statistics.linearmodels_stats import LinearmodelsStats
from lmdiag.statistics.sklearn_stats import SklearnStats
from lmdiag.statistics.statsmodels_stats import StatsmodelsStats


@pytest.mark.parametrize("x_dims", [1, 3, 5])
@pytest.mark.parametrize(
    "attr",
    [
        "residuals",
        "fitted_values",
        "standard_residuals",
        "cooks_d",
        "leverage",
        "parameter_count",
    ],
)
def test_lm_stats_modules(
    attr: str,
    statsmodels_factory: Callable,
    linearmodels_factory: Callable,
    sklearn_factory: Callable,
    x_dims: int,
) -> None:
    model_stats_to_compare = [
        StatsmodelsStats(statsmodels_factory(x_dims=x_dims)),
        LinearmodelsStats(linearmodels_factory(x_dims=x_dims)),
        SklearnStats(*sklearn_factory(x_dims=x_dims)),
    ]

    for stats_a, stats_b in itertools.combinations(model_stats_to_compare, 2):
        vector_a = getattr(stats_a, attr)
        vector_b = getattr(stats_b, attr)

        assert type(vector_a) == type(vector_b), (
            attr,
            stats_a.__class__.__name__,
            stats_b.__class__.__name__,
        )

        assert np.allclose(vector_a, vector_b, atol=1e-10), (
            attr,
            stats_a.__class__.__name__,
            stats_b.__class__.__name__,
        )
