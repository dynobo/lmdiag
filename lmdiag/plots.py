"""Module for Diagnosis Plots of Linear Regression Models."""

from typing import Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

from lmdiag.lm_stats.base import StatsBase

try:
    import linearmodels
except ImportError:
    linearmodels = None

TITLE_SIZE = 15
EDGE_COLOR = (0, 0, 0, 0.6)

LOWESS_DELTA = 0.005
LOWESS_IT = 2


def init_lm_stats(lm: Any) -> StatsBase:
    """Check if input parameter is an linear regression model."""
    if isinstance(lm, sm.regression.linear_model.RegressionResultsWrapper):
        from lmdiag.lm_stats.statsmodels import StatsmodelsStats

        return StatsmodelsStats(lm)

    if linearmodels and isinstance(lm, linearmodels.iv.results.OLSResults):
        from lmdiag.lm_stats.linearmodels import LinearmodelsStats

        return LinearmodelsStats(lm)

    raise TypeError("Model type not (yet) supported.")


def resid_fit(
    lm: Any,
    ax: Optional[mpl.axes.Axes] = None,
    lowess_delta: float = LOWESS_DELTA,
    lowess_it: int = LOWESS_IT,
) -> mpl.figure.Figure:
    """Draw Residuals vs. Fitted Values Plot."""
    lm_stats = lm if isinstance(lm, StatsBase) else init_lm_stats(lm)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Calculate values for scatter points
    fitted = lm_stats.fitted_values
    residuals = lm_stats.residuals

    # Calculate lowess for smoothing line
    grid, yhat = lowess(residuals, fitted, it=lowess_it, delta=lowess_delta).T

    # Get top three observations for annotation
    top_3 = np.abs(residuals).argsort()[-3:][::1]

    # Draw scatter and lowess line
    ax.plot([fitted.min(), fitted.max()], [0, 0], "k:")
    ax.plot(grid, yhat, "r-")
    ax.plot(fitted, residuals, "o", mec=EDGE_COLOR, markeredgewidth=1, fillstyle="none")

    # Draw Annotations
    for point in top_3:
        ax.annotate(point, xy=(fitted[point], residuals[point]), color="r")

    # Set Labels
    ax.set_title("Residual vs. Fitted", fontsize=TITLE_SIZE)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")

    return fig


def q_q(lm: Any, ax: Optional[mpl.axes.Axes] = None) -> mpl.axes.Axes:
    """Draw Q-Q-Plot."""
    lm_stats = lm if isinstance(lm, StatsBase) else init_lm_stats(lm)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Calculate values for scatter points
    std_resid = lm_stats.standard_residuals
    quantiles = lm_stats.normalized_quantiles

    # Sort for Q-Q plot
    std_resid_sort = np.sort(std_resid)
    quantiles_sort = np.sort(quantiles)

    # Function for fitted line
    fit = np.polyfit(quantiles_sort, std_resid_sort, deg=1)

    # Get top three observations for annotation
    # (need position of sorted for coord, and original for label)
    top_3_sorted = np.abs(std_resid_sort).argsort()[-3:][::1]
    top_3_orig = np.abs(std_resid).argsort()[-3:][::1]
    top_3 = zip(top_3_sorted, top_3_orig)

    ax.plot(quantiles_sort, fit[0] * quantiles_sort + fit[1], "r:")
    ax.plot(
        quantiles_sort,
        std_resid_sort,
        "o",
        mec=EDGE_COLOR,
        markeredgewidth=1,
        mfc="none",
    )

    for point in top_3:
        ax.annotate(
            point[1], xy=(quantiles_sort[point[0]], std_resid_sort[point[0]]), color="r"
        )

    ax.set_title("Normal Q-Q", fontsize=TITLE_SIZE)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Standardized residuals")

    return fig


def scale_loc(
    lm: Any,
    ax: Optional[mpl.axes.Axes] = None,
    lowess_delta: float = LOWESS_DELTA,
    lowess_it: int = LOWESS_IT,
) -> mpl.figure.Figure:
    """Draw Scale-Location Plot."""
    lm_stats = lm if isinstance(lm, StatsBase) else init_lm_stats(lm)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Get Fitted Values
    fitted_vals = lm_stats.fitted_values
    sqrt_abs_res = lm_stats.sqrt_abs_residuals

    # Get top three observations for annotation
    top_3 = sqrt_abs_res.argsort()[-3:][::1]

    # Calculate lowess for smoothing line
    grid, yhat = lowess(sqrt_abs_res, fitted_vals, it=lowess_it, delta=lowess_delta).T

    ax.plot(grid, yhat, "r-")
    ax.plot(
        fitted_vals,
        sqrt_abs_res,
        "o",
        mec=EDGE_COLOR,
        markeredgewidth=1,
        fillstyle="none",
    )

    for point in top_3:
        ax.annotate(point, xy=(fitted_vals[point], sqrt_abs_res[point]), color="r")

    ax.set_title("Scale-Location", fontsize=TITLE_SIZE)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel(r"$\sqrt{\left|\mathregular{Standardized\ residuals}\right|}$")

    return fig


def resid_lev(
    lm: Any,
    ax: Optional[mpl.axes.Axes] = None,
    lowess_delta: float = LOWESS_DELTA,
    lowess_it: int = LOWESS_IT,
) -> mpl.figure.Figure:
    """Draw Standardized Residuals vs. Leverage Plot."""
    lm_stats = lm if isinstance(lm, StatsBase) else init_lm_stats(lm)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    std_resid = lm_stats.standard_residuals
    cooks_d = lm_stats.cooks_d
    leverage = lm_stats.leverage

    # Get top three observations for annotation
    top_3 = cooks_d.argsort()[-3:][::1]

    # Get Cooks Distance contour lines
    x = np.linspace(leverage.min(), leverage.max(), 100)
    params_count = lm_stats.params_count

    # Calculate lowess for smoothing line
    grid, yhat = lowess(std_resid, leverage, it=lowess_it, delta=lowess_delta).T

    # Draw cooks distance contours
    ax.plot(x, np.sqrt((0.5 * params_count * (1 - x)) / x), "r--")
    ax.plot(x, np.sqrt((1.0 * params_count * (1 - x)) / x), "r--")
    ax.plot(x, np.negative(np.sqrt((0.5 * params_count * (1 - x)) / x)), "r--")
    ax.plot(x, np.negative(np.sqrt((1.0 * params_count * (1 - x)) / x)), "r--")

    # Draw lowess line
    ax.plot(grid, yhat, "r-")

    # Draw data points
    ax.plot(
        leverage, std_resid, "o", mec=EDGE_COLOR, markeredgewidth=1, fillstyle="none"
    )

    # Limit y axis to actual values (otherwise contour lines disturb scale)
    ax.set_ylim(std_resid.min() * 1.1, std_resid.max() * 1.1)

    # Draw data point annotations
    for point in top_3:
        ax.annotate(point, xy=(leverage[point], std_resid[point]), color="r")

    ax.set_title("Residuals vs. Leverage", fontsize=TITLE_SIZE)
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized residuals")

    return fig


def plot(
    lm: Any, lowess_delta: float = LOWESS_DELTA, lowess_it: int = LOWESS_IT
) -> mpl.figure.Figure:
    """Plot all 4 charts as a Matrix."""
    lm_stats = lm if isinstance(lm, StatsBase) else init_lm_stats(lm)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    resid_fit(lm_stats, ax=axs[0][0], lowess_delta=lowess_delta, lowess_it=lowess_it)
    q_q(lm_stats, ax=axs[0][1])
    scale_loc(lm_stats, ax=axs[1][0], lowess_delta=lowess_delta, lowess_it=lowess_it)
    resid_lev(lm_stats, ax=axs[1][1], lowess_delta=lowess_delta, lowess_it=lowess_it)

    fig.tight_layout(pad=0.5, w_pad=4, h_pad=3.5)

    return fig


if __name__ == "__main__":
    # Example used for debugging
    df = sm.datasets.get_rdataset("ames", "openintro").data
    y = np.log10(df["price"])
    x = df["Overall.Qual"] + np.log(df["area"])
    x = sm.add_constant(x)

    lm = sm.OLS(y, x).fit()
    fig = plot(lm)
    fig.savefig("test.png")
