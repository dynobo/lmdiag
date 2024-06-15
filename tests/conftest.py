from typing import Callable

import numpy as np
import pytest
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

SAMPLE_DATA = sm.datasets.longley.load()


def _get_sample_data(x_dims: int) -> tuple[np.ndarray, np.ndarray]:
    y = SAMPLE_DATA.endog.values
    X = SAMPLE_DATA.exog.values[:, :x_dims]
    return X, y


@pytest.fixture(scope="session")
def statsmodels_factory() -> Callable:
    def _statsmodels_lm(x_dims: int) -> sm.OLS:
        X, y = _get_sample_data(x_dims=x_dims)
        X = sm.add_constant(X)
        return sm.OLS(y, X).fit()

    return _statsmodels_lm


@pytest.fixture(scope="session")
def linearmodels_factory() -> Callable:
    def _linearmodels_lm(x_dims: int) -> IV2SLS:
        X, y = _get_sample_data(x_dims=x_dims)
        X = sm.add_constant(X)
        return IV2SLS(y, X, None, None).fit(cov_type="unadjusted")

    return _linearmodels_lm
