from typing import Callable

import numpy as np
import pytest
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from sklearn.linear_model import LinearRegression

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
        lm = sm.OLS(y, X).fit()
        return lm

    return _statsmodels_lm


@pytest.fixture(scope="session")
def linearmodels_factory() -> Callable:
    def _linearmodels_lm(x_dims: int) -> IV2SLS:
        X, y = _get_sample_data(x_dims=x_dims)
        X = sm.add_constant(X)
        lm = IV2SLS(y, X, None, None).fit(cov_type="unadjusted")
        return lm

    return _linearmodels_lm


@pytest.fixture(scope="session")
def sklearn_factory() -> Callable:
    def _sklearn_lm(x_dims: int) -> tuple[Callable, np.ndarray, np.ndarray]:
        X, y = _get_sample_data(x_dims=x_dims)
        lm = LinearRegression().fit(X, y)
        return lm, X, y

    return _sklearn_lm
