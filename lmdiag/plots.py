"""Module for diagnosis plots of linear regression models."""

from typing import Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

from lmdiag import style
from lmdiag.statistics.base import StatsBase
from lmdiag.statistics.select import get_stats

LOWESS_DELTA = 0.005
LOWESS_IT = 2


def _get_figure(
    ax: Optional[mpl.axes.Axes] = None,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Retrieve figure of axes or create a new one."""
    if ax is None:
        fig, ax = plt.subplots(**style.subplots)
    else:
        fig = ax.get_figure()
    return fig, ax


def resid_fit(
    lm: Any,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    ax: Optional[mpl.axes.Axes] = None,
    lowess_delta: float = LOWESS_DELTA,
    lowess_it: int = LOWESS_IT,
) -> mpl.figure.Figure:
    """Draw Residuals vs. Fitted Values Plot.

    For detailed explanation fo the lowess parameters, see:
    https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html

    Args:
        lm: A fitted linear model of a supported type.
        x: X (predictor) of training data. Only for `sklearn` models!
        y: y (true response) of training data. Only for `sklearn` models!
        ax: Matplotlib axes for drawing. If `None` (default), a new Figure and Axes are
            created.
        lowess_delta: A value between 0 and 1. Higher values speed up plotting, but
            reduce the accuracy of the smoothing line. Defaults to 0.005. See:
        lowess_it: Lower values speed up plotting, but reduce the accuracy of the
            smoothing line. Defaults to 2.

    Returns:
        Figure of the plot.
    """
    lm_stats = lm if isinstance(lm, StatsBase) else get_stats(lm, x=x, y=y)

    fitted = lm_stats.fitted_values
    residuals = lm_stats.residuals
    smooth_x, smooth_y = lowess(residuals, fitted, it=lowess_it, delta=lowess_delta).T
    top_3_observations = np.abs(residuals).argsort()[-3:][::1]

    fig, ax = _get_figure(ax=ax)
    ax.plot([fitted.min(), fitted.max()], [0, 0], **style.plot_contour)
    ax.plot(smooth_x, smooth_y, **style.plot)
    ax.scatter(fitted, residuals, **style.scatter)
    for ob in top_3_observations:
        ax.annotate(ob, xy=(fitted[ob], residuals[ob]), **style.annotate)
    ax.set_title("Residual vs. Fitted", **style.title)
    ax.set_xlabel("Fitted values", **style.xy_label)
    ax.set_ylabel("Residuals", **style.xy_label)

    return fig


def q_q(
    lm: Any,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    ax: Optional[mpl.axes.Axes] = None,
) -> mpl.figure.Figure:
    """Draw Q-Q-Plot.

    Args:
        lm: A fitted linear model of a supported type.
        x: X (predictor) of training data. Only for `sklearn` models!
        y: y (true response) of training data. Only for `sklearn` models!
        ax: Matplotlib axes for drawing. If `None` (default), a new Figure and Axes are
            created.

    Returns:
        Figure of the plot.
    """
    lm_stats = lm if isinstance(lm, StatsBase) else get_stats(lm, x=x, y=y)

    std_resid = lm_stats.standard_residuals
    quantiles = lm_stats.normalized_quantiles
    std_resid_sorted = np.sort(std_resid)
    quantiles_sorted = np.sort(quantiles)
    fitted_values = np.polyfit(quantiles_sorted, std_resid_sorted, deg=1)
    fitted_line_y = fitted_values[0] * quantiles_sorted + fitted_values[1]

    # Index of sorted obs is used for annotation position, original index as label:
    top_3_sorted = np.abs(std_resid_sorted).argsort()[-3:][::1]
    top_3_orig = np.abs(std_resid).argsort()[-3:][::1]
    top_3_observations = zip(top_3_sorted, top_3_orig)

    fig, ax = _get_figure(ax=ax)
    ax.plot(quantiles_sorted, fitted_line_y, ":", **style.plot)
    ax.scatter(quantiles_sorted, std_resid_sorted, **style.scatter)
    for ob in top_3_observations:
        ax.annotate(
            ob[1],
            xy=(quantiles_sorted[ob[0]], std_resid_sorted[ob[0]]),
            **style.annotate,
        )
    ax.set_title("Normal Q-Q", **style.title)
    ax.set_xlabel("Theoretical Quantiles", **style.xy_label)
    ax.set_ylabel("Standardized residuals", **style.xy_label)

    return fig


def scale_loc(
    lm: Any,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    ax: Optional[mpl.axes.Axes] = None,
    lowess_delta: float = LOWESS_DELTA,
    lowess_it: int = LOWESS_IT,
) -> mpl.figure.Figure:
    """Draw Scale-Location Plot.

    For detailed explanation fo the lowess parameters, see:
    https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html

    Args:
        lm: A fitted linear model of a supported type.
        x: X (predictor) of training data. Only for `sklearn` models!
        y: y (true response) of training data. Only for `sklearn` models!
        ax: Matplotlib axes for drawing. If `None` (default), a new Figure and Axes are
            created.
        lowess_delta: A value between 0 and 1. Higher values speed up plotting, but
            reduce the accuracy of the smoothing line. Defaults to 0.005. See:
        lowess_it: Lower values speed up plotting, but reduce the accuracy of the
            smoothing line. Defaults to 2.

    Returns:
        Figure of the plot.
    """
    lm_stats = lm if isinstance(lm, StatsBase) else get_stats(lm, x=x, y=y)

    fitted_vals = lm_stats.fitted_values
    sqrt_abs_res = lm_stats.sqrt_abs_residuals
    top_3_observations = sqrt_abs_res.argsort()[-3:][::1]
    smooth_x, smooth_y = lowess(
        sqrt_abs_res, fitted_vals, it=lowess_it, delta=lowess_delta
    ).T

    fig, ax = _get_figure(ax=ax)
    ax.plot(smooth_x, smooth_y, **style.plot)
    ax.scatter(fitted_vals, sqrt_abs_res, **style.scatter)
    for ob in top_3_observations:
        ax.annotate(ob, xy=(fitted_vals[ob], sqrt_abs_res[ob]), **style.annotate)
    ax.set_title("Scale-Location", **style.title)
    ax.set_xlabel("Fitted values", **style.xy_label)
    ax.set_ylabel(
        r"$\sqrt{\left|\mathregular{Standardized\ residuals}\right|}$", **style.xy_label
    )

    return fig


def resid_lev(
    lm: Any,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    ax: Optional[mpl.axes.Axes] = None,
    lowess_delta: float = LOWESS_DELTA,
    lowess_it: int = LOWESS_IT,
) -> mpl.figure.Figure:
    """Draw Standardized Residuals vs. Leverage Plot.

    For detailed explanation fo the lowess parameters, see:
    https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html

    Args:
        lm: A fitted linear model of a supported type.
        x: X (predictor) of training data. Only for `sklearn` models!
        y: y (true response) of training data. Only for `sklearn` models!
        ax: Matplotlib axes for drawing. If `None` (default), a new Figure and Axes are
            created.
        lowess_delta: A value between 0 and 1. Higher values speed up plotting, but
            reduce the accuracy of the smoothing line. Defaults to 0.005. See:
        lowess_it: Lower values speed up plotting, but reduce the accuracy of the
            smoothing line. Defaults to 2.

    Returns:
        Figure of the plot.
    """
    lm_stats = lm if isinstance(lm, StatsBase) else get_stats(lm, x=x, y=y)

    std_resid = lm_stats.standard_residuals
    cooks_d = lm_stats.cooks_d
    leverage = lm_stats.leverage
    top_3_observations = cooks_d.argsort()[-3:][::1]
    x_ = np.linspace(leverage.min(), leverage.max(), 100)
    params_count = lm_stats.parameter_count
    smooth_x, smooth_y = lowess(std_resid, leverage, it=lowess_it, delta=lowess_delta).T

    fig, ax = _get_figure(ax=ax)

    # Draw cooks distance contours
    ax.plot(x_, np.sqrt((0.5 * params_count * (1 - x_)) / x_), **style.plot_contour)
    ax.plot(x_, np.sqrt((1.0 * params_count * (1 - x_)) / x_), **style.plot_contour)
    ax.plot(
        x_,
        np.negative(np.sqrt((0.5 * params_count * (1 - x_)) / x_)),
        **style.plot_contour,
    )
    ax.plot(
        x_,
        np.negative(np.sqrt((1.0 * params_count * (1 - x_)) / x_)),
        **style.plot_contour,
    )

    ax.plot(smooth_x, smooth_y, **style.plot)
    ax.scatter(leverage, std_resid, **style.scatter)

    # Limit y axis to actual values (otherwise contour lines disturb scale)
    ax.set_ylim(std_resid.min() * 1.1, std_resid.max() * 1.1)

    for ob in top_3_observations:
        ax.annotate(ob, xy=(leverage[ob], std_resid[ob]), **style.annotate)

    ax.set_title("Residuals vs. Leverage", **style.title)
    ax.set_xlabel("Leverage", **style.xy_label)
    ax.set_ylabel("Standardized residuals", **style.xy_label)

    return fig


def plot(
    lm: Any,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    lowess_delta: float = LOWESS_DELTA,
    lowess_it: int = LOWESS_IT,
) -> mpl.figure.Figure:
    """Plot all 4 charts as a Matrix.

    For detailed explanation fo the lowess parameters, see:
    https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html

    Args:
        lm: A fitted linear model of a supported type.
        x: X (predictor) of training data. Only for `sklearn` models!
        y: y (true response) of training data. Only for `sklearn` models!
        lowess_delta: A value between 0 and 1. Higher values speed up plotting, but
            reduce the accuracy of the smoothing line. Defaults to 0.005. See:
        lowess_it: Lower values speed up plotting, but reduce the accuracy of the
            smoothing line. Defaults to 2.

    Returns:
        Figure of the plot.
    """
    lm_stats = get_stats(lm=lm, x=x, y=y)

    fig, axs = plt.subplots(2, 2, **style.subplots)

    resid_fit(lm_stats, ax=axs[0][0], lowess_delta=lowess_delta, lowess_it=lowess_it)
    q_q(lm_stats, ax=axs[0][1])
    scale_loc(lm_stats, ax=axs[1][0], lowess_delta=lowess_delta, lowess_it=lowess_it)
    resid_lev(lm_stats, ax=axs[1][1], lowess_delta=lowess_delta, lowess_it=lowess_it)

    fig.tight_layout(**style.tight_layout)

    return fig


if __name__ == "__main__":
    # Example used for debugging
    df = sm.datasets.get_rdataset("ames", "openintro").data
    y = np.log10(df["price"])
    X = df["Overall.Qual"] + np.log(df["area"])
    X = sm.add_constant(X)

    lm = sm.OLS(y, X).fit()
    fig = plot(lm)
    fig.savefig("test.png")
