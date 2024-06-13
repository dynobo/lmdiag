from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pytest
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

from lmdiag import plot


def get_predictor_response() -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(20)
    predictor = np.random.normal(size=30, loc=20, scale=3)
    response = 5 + 5 * predictor + np.random.normal(size=30)
    return predictor, response


def fitted_statsmodels_lm() -> sm.OLS:
    predictor, response = get_predictor_response()
    x = sm.add_constant(predictor)
    return sm.OLS(response, x).fit()


def fitted_linearmodels_lm() -> IV2SLS:
    predictor, response = get_predictor_response()
    x = sm.add_constant(predictor)
    return IV2SLS(response, x, None, None).fit(cov_type="unadjusted")


@pytest.mark.parametrize(
    ("lm_type", "lm_fitted"),
    [
        ("statsmodels", fitted_statsmodels_lm()),
        ("linearmodels", fitted_linearmodels_lm()),
    ],
)
def test_plot_generates_expected_image(
    lm_type: str, lm_fitted: Union[sm.OLS, IV2SLS]
) -> None:
    base_path = Path(__file__).parent
    filename = base_path / f"test_plot_actual_{lm_type}.jpg"

    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(10, 7))

    # Using lowess_delta leads to barely visible differences between plot of
    # linearmodels and statsmodels, therefore we set it to 0 during testing
    fig = plot(lm_fitted, lowess_delta=0)
    fig.savefig(filename)

    actual_size = Path(filename).stat().st_size
    expected_size = Path(base_path / "test_plot_expected.jpg").stat().st_size

    assert actual_size == expected_size, lm_type
