from typing import TypedDict, Union


class MplKwargs(TypedDict, total=False):
    """Types for the most common matplotlib arguments to please mypy."""

    alpha: Union[float, None]
    c: Union[str, tuple]
    color: Union[str, tuple]
    edgecolors: Union[str, tuple]
    fillstyle: str
    fontsize: int
    fontweight: str
    linestyle: str
    linewidth: float
    marker: str
    markeredgecolor: Union[str, tuple]
    markerfacecolor: Union[str, tuple]
    markersize: float
    visible: bool
    zorder: float
    pad: float
    h_pad: float
    w_pad: float
    figsize: tuple


scatter: MplKwargs = {"alpha": 0.5, "color": "C0"}
plot: MplKwargs = {"color": "C1"}
plot_contour: MplKwargs = {"color": "dimgrey", "linestyle": "dashed"}
xy_label: MplKwargs = {}
annotate: MplKwargs = {"color": plot["color"]}
title: MplKwargs = {}
tight_layout: MplKwargs = {"h_pad": 2.5, "w_pad": 3}
subplots: MplKwargs = {}


def use(style: str) -> None:
    """Set predefined style for plots.

    Available styles:
    - 'black_and_red' (mimics style of R's lm.diag)

    Args:
        style: Name of the preset style.

    Raises:
        ValueError: If style is unknown.
    """
    if style == "black_and_red":
        scatter.update(
            {"marker": "o", "color": "none", "edgecolors": "black", "linewidth": 1}
        )
        plot.update({"color": "red"})
        plot_contour.update({"color": "dimgrey"})
        annotate.update({"color": plot["color"]})
        subplots.update({"figsize": (10, 7)})
        title.update({"fontsize": 15})
    else:
        raise ValueError(f"Unknown style '{style}'.")
