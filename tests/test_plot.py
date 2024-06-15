from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from lmdiag import plot


@pytest.mark.parametrize(
    "lm",
    ["statsmodels_lm", "linearmodels_lm", "sklearn_lm"],
)
def test_plot_generates_expected_image(lm: str, request: pytest.FixtureRequest) -> None:
    base_path = Path(__file__).parent
    filename = base_path / f"test_plot_actual_{lm}.jpg"

    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(10, 7))

    lm_fitted = request.getfixturevalue(lm)
    # Using lowess_delta leads to barely visible differences between plot of
    # linearmodels and statsmodels, therefore we set it to 0 during testing
    fig = plot(lm_fitted, lowess_delta=0)
    fig.savefig(filename)

    actual_size = Path(filename).stat().st_size
    expected_size = Path(base_path / "test_plot_expected.jpg").stat().st_size

    assert actual_size == expected_size
