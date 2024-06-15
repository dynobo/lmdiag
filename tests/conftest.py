import numpy as np
import pytest
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

def _get_predictor_response() -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(20)
    predictor = np.random.normal(size=30, loc=20, scale=3)
    response = 5 + 5 * predictor + np.random.normal(size=30)

    X = np.column_stack((predictor, predictor**2))
    return X, response


@pytest.fixture(scope="session")
def statsmodels_lm() -> sm.OLS:
    predictor, response = _get_predictor_response()
    x = sm.add_constant(predictor)
    return sm.OLS(response, x).fit()


@pytest.fixture(scope="session")
def linearmodels_lm() -> IV2SLS:
    predictor, response = _get_predictor_response()
    x = sm.add_constant(predictor)
    return IV2SLS(response, x, None, None).fit(cov_type="unadjusted")

