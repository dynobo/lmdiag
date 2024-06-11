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
    "fitted_lm", [fitted_statsmodels_lm(), fitted_linearmodels_lm()]
)
def test_plot_generates_expected_image(
    fitted_lm: Union[sm.OLS, IV2SLS], tmpdir: Path
) -> None:
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(10, 7))
    fig = plot(fitted_lm)

    fig.savefig(tmpdir / "test_plot_actual.jpg")
    actual_size = Path(tmpdir / "test_plot_actual.jpg").stat().st_size

    expected_size = (Path(__file__).parent / "test_plot_expected.jpg").stat().st_size

    assert actual_size == expected_size
