"""Module for Diagnosis Plots of Linear Regression Models."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

from lmdiag.lm_stats.base import StatsBase
from lmdiag.lm_stats.linearmodels import LinearmodelsStats
from lmdiag.lm_stats.statsmodels import StatsmodelsStats

try:
    import linearmodels
except ImportError:
    linearmodels = None

# GLOBAL STYLE VARIABLES
title_size = 15
edge_col = (0, 0, 0, 0.6)


def init_model_stats(lm: Any) -> StatsBase:
    """Check if input parameter is an linear regression model."""
    if isinstance(lm, sm.regression.linear_model.RegressionResultsWrapper):
        return StatsmodelsStats(lm)
    if linearmodels and isinstance(lm, linearmodels.iv.results.OLSResults):
        return LinearmodelsStats(lm)
    raise TypeError("Model type not (yet) supported.")


def resid_fit(lm: Any) -> plt:
    """Draw Residuals vs. Fitted Values Plot."""
    lm_stats = init_model_stats(lm)

    # Calculate values for scatter points
    fitted = lm_stats.fitted_values
    residuals = lm_stats.residuals

    # Calculate lowess for smoothing line
    grid, yhat = lowess(residuals, fitted).T

    # Get top three observations for annotation
    top_3 = np.abs(residuals).argsort()[-3:][::1]

    # Draw scatter and lowess line
    plt.plot([fitted.min(), fitted.max()], [0, 0], "k:")
    plt.plot(grid, yhat, "r-")
    plt.plot(fitted, residuals, "o", mec=edge_col, markeredgewidth=1, fillstyle="none")

    # Draw Annotations
    for point in top_3:
        plt.annotate(point, xy=(fitted[point], residuals[point]), color="r")

    # Set Labels
    plt.title("Residual vs. Fitted", fontsize=title_size)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")

    return plt


def q_q(lm: Any) -> plt:
    """Draw Q-Q-Plot."""
    lm_stats = init_model_stats(lm)

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

    # Draw scatter and fitted line
    plt.plot(quantiles_sort, fit[0] * quantiles_sort + fit[1], "r:")
    plt.plot(
        quantiles_sort, std_resid_sort, "o", mec=edge_col, markeredgewidth=1, mfc="none"
    )

    # Draw Annotations
    for point in top_3:
        plt.annotate(
            point[1], xy=(quantiles_sort[point[0]], std_resid_sort[point[0]]), color="r"
        )

    # Set Labels
    plt.title("Normal Q-Q", fontsize=title_size)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Standardized residuals")

    return plt


def scale_loc(lm: Any) -> plt:
    """Draw Scale-Location Plot."""
    lm_stats = init_model_stats(lm)

    # Get Fitted Values
    fitted_vals = lm_stats.fitted_values
    sqrt_abs_res = lm_stats.sqrt_abs_residuals

    # Get top three observations for annotation
    top_3 = sqrt_abs_res.argsort()[-3:][::1]

    # Calculate lowess for smoothing line
    grid, yhat = lowess(sqrt_abs_res, fitted_vals).T

    # Draw scatter and lowess line
    plt.plot(grid, yhat, "r-")
    plt.plot(
        fitted_vals,
        sqrt_abs_res,
        "o",
        mec=edge_col,
        markeredgewidth=1,
        fillstyle="none",
    )

    # Draw Annotations
    for point in top_3:
        plt.annotate(point, xy=(fitted_vals[point], sqrt_abs_res[point]), color="r")

    # Set Labels
    plt.title("Scale-Location", fontsize=title_size)
    plt.xlabel("Fitted values")
    plt.ylabel(r"$\sqrt{\left|Standardized\ residuals\right|}$")

    return plt


def resid_lev(lm: Any) -> plt:
    """Draw Standardized Residuals vs. Leverage Plot."""
    lm_stats = init_model_stats(lm)

    std_resid = lm_stats.standard_residuals
    cooks_d = lm_stats.cooks_d
    leverage = lm_stats.leverage

    # Get top three observations for annotation
    top_3 = cooks_d.argsort()[-3:][::1]

    # Get Cooks Distance contour lines
    x = np.linspace(leverage.min(), leverage.max(), 100)
    params_count = lm_stats.params_count

    # Calculate lowess for smoothing line
    grid, yhat = lowess(std_resid, leverage).T

    # Draw cooks distance contours, scatter and lowess line
    plt.plot(x, np.sqrt((0.5 * params_count * (1 - x)) / x), "r--")
    plt.plot(x, np.sqrt((1.0 * params_count * (1 - x)) / x), "r--")
    plt.plot(x, np.negative(np.sqrt((0.5 * params_count * (1 - x)) / x)), "r--")
    plt.plot(x, np.negative(np.sqrt((1.0 * params_count * (1 - x)) / x)), "r--")
    plt.plot(grid, yhat, "r-")
    plt.plot(
        leverage, std_resid, "o", mec=edge_col, markeredgewidth=1, fillstyle="none"
    )

    # Limit y axis to actual values (otherwise contour lines disturb scale)
    plt.ylim(std_resid.min() * 1.1, std_resid.max() * 1.1)

    # Draw Annotations
    for point in top_3:
        plt.annotate(point, xy=(leverage[point], std_resid[point]), color="r")

    # Set Labels
    plt.title("Residuals vs. Leverage", fontsize=title_size)
    plt.xlabel("Leverage")
    plt.ylabel("Standardized residuals")

    return plt


def plot(lm: Any) -> plt:
    """Plot all 4 charts as a Matrix."""
    # Draw plot by plot
    plt.subplot(2, 2, 1)
    resid_fit(lm)

    plt.subplot(2, 2, 2)
    q_q(lm)

    plt.subplot(2, 2, 3)
    scale_loc(lm)

    plt.subplot(2, 2, 4)
    resid_lev(lm)

    # Padding between Charts
    plt.tight_layout(pad=0.5, w_pad=4, h_pad=4)

    return plt
