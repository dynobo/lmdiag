from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pytest

from lmdiag import plot


@pytest.mark.parametrize(
    "lm_type",
    ["statsmodels", "linearmodels", "sklearn"],
)
def test_plot_generates_expected_image(
    lm_type: str,
    statsmodels_factory: Callable,
    linearmodels_factory: Callable,
    sklearn_factory: Callable,
) -> None:
    base_path = Path(__file__).parent
    filename = base_path / f"test_plot_actual_{lm_type}.jpg"

    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(10, 7))

    if lm_type == "statsmodels":
        lm_fitted = statsmodels_factory(x_dims=5)
    elif lm_type == "linearmodels":
        lm_fitted = linearmodels_factory(x_dims=5)
    elif lm_type == "sklearn":
        lm_fitted = sklearn_factory(x_dims=5)
    else:
        raise ValueError(f"Unsupported lm_type: {lm_type}")

    # Using lowess_delta leads to small differences between plot of
    # linearmodels and statsmodels, therefore we set it to 0 during testing
    fig = plot(lm_fitted, lowess_delta=0)
    fig.savefig(filename)

    # Quick and simple way to compare the plots is to compare jpg file sizes
    actual_bytes = Path(filename).stat().st_size
    expected_bytes = Path(base_path / "test_plot_expected.jpg").stat().st_size
    acceptable_byte_difference = 10
    assert abs(actual_bytes - expected_bytes) < acceptable_byte_difference
