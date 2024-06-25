from typing import Callable

import matplotlib as mpl
import pytest

import lmdiag


@pytest.mark.parametrize("style", ["black_and_red", "default"])
def test_use(style: str, statsmodels_factory: Callable) -> None:
    lm = statsmodels_factory(x_dims=2)
    lmdiag.style.use(style)
    fig = lmdiag.plot(lm=lm)
    assert isinstance(fig, mpl.figure.Figure)
