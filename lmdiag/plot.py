"""Module for Diagnosis Plots of Lineare Regression Models."""

# Extra Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table, OLSInfluence
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import norm

# Global Style Variables
title_size = 15
edge_col = (0, 0, 0, 0.6)


# VERIFICATIONS
# ----------------

def verify_input(lm):
    """Check if input parameter is an linear regression model."""
    if not isinstance(lm, sm.regression.linear_model.RegressionResultsWrapper):
        raise TypeError('Input is no statsmodel OLS model!')
    return


# GATHERING VALUES
# ----------------

def get_residuals(lm):
    tbl, data, labels = summary_table(lm, alpha=0.05)
    residuals = data[:, 8]
    return residuals


def get_fitted_values(lm):
    """Return 1-D numpy array with fitted values."""
    fitted = lm.fittedvalues
    # Transform series to 1-d array, if necessary
    if isinstance(fitted, pd.Series):
        fitted = fitted.values
    return fitted


def get_standard_residuals(lm):
    vals = OLSInfluence(lm).summary_frame()
    std_resid = vals['standard_resid'].values
    return std_resid


def get_sqrt_abs_residuals(lm):
    """Return sqrt(|Standardized resiudals|)."""
    std_resid = get_standard_residuals(lm)
    sqrt_abs_res = np.sqrt(np.abs(std_resid))
    return sqrt_abs_res


def get_normalized_quantiles(lm):
    val_count = len(lm.fittedvalues)
    positions = (np.arange(1., val_count + 1))/(val_count + 1)
    norm_quantiles = norm.ppf(positions)
    return norm_quantiles


def get_cooks_d(lm):
    vals = OLSInfluence(lm).summary_frame()
    cooks_d = vals['cooks_d'].values
    return cooks_d


# DRAWING CHARTS
# ---------------

def resid_fit(lm):
    """Draw Residuals vs. Fitted Values Plot."""
    verify_input(lm)

    # Calculate values for scatter points
    fitted = get_fitted_values(lm)
    residuals = get_residuals(lm)

    # Calculate lowess for smoothing line
    grid, yhat = lowess(residuals, fitted).T

    # Get top three observations for annotation
    top_3 = np.abs(residuals).argsort()[-3:][::1]

    # Draw scatter and lowess line
    plt.plot([fitted.min(), fitted.max()], [0, 0], 'k:')
    plt.plot(grid, yhat, 'r-')
    plt.plot(fitted,
             residuals,
             'o',
             mec=edge_col,
             markeredgewidth=1,
             fillstyle='none')

    # Draw Annotations
    for point in top_3:
        plt.annotate(point,
                     xy=(fitted[point], residuals[point]),
                     color='r')

    # Set Labels
    plt.title('Residual vs. Fitted', fontsize=title_size)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')

    return plt


def q_q(lm):
    """Draw Q-Q-Plot."""
    verify_input(lm)

    # Calulate values for scatter points
    std_resid = get_standard_residuals(lm)
    quantiles = get_normalized_quantiles(lm)

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
    plt.plot(quantiles_sort, fit[0] * quantiles_sort + fit[1], 'r:')
    plt.plot(quantiles_sort,
             std_resid_sort,
             'o',
             mec=edge_col,
             markeredgewidth=1,
             mfc='none')

    # Draw Annotations
    for point in top_3:
        plt.annotate(point[1],
                     xy=(quantiles_sort[point[0]], std_resid_sort[point[0]]),
                     color='r')

    # Set Labels
    plt.title('Normal Q-Q', fontsize=title_size)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Standardized residuals')

    return plt


def scale_loc(lm):
    """Draw Scale-Location Plot."""
    verify_input(lm)

    # Get Fitted Values
    fitted_vals = get_fitted_values(lm)
    sqrt_abs_res = get_sqrt_abs_residuals(lm)

    # Get top three observations for annotation
    top_3 = sqrt_abs_res.argsort()[-3:][::1]

    # Calculate lowess for smoothing line
    grid, yhat = lowess(sqrt_abs_res, fitted_vals).T

    # Draw scatter and lowess line
    plt.plot(grid, yhat, 'r-')
    plt.plot(fitted_vals,
             sqrt_abs_res,
             'o',
             mec=edge_col,
             markeredgewidth=1,
             fillstyle='none')

    # Draw Annotations
    for point in top_3:
        plt.annotate(point,
                     xy=(fitted_vals[point], sqrt_abs_res[point]),
                     color='r')

    # Set Labels
    plt.title('Scale-Location', fontsize=title_size)
    plt.xlabel('Fitted values')
    plt.ylabel(r'$\sqrt{\left|Standardized\ residuals\right|}$')

    return plt


def resid_lev(lm):
    """Draw Stanardized Residuals vs. Leverage Plot."""
    verify_input(lm)

    # Get stanardized residuals & cooks distance
    std_resid = get_standard_residuals(lm)
    cooks_d = get_cooks_d(lm)

    # Get top three observations for annotation
    top_3 = cooks_d.argsort()[-3:][::1]

    # Get Leverage
    infl = lm.get_influence()
    leverage = infl.hat_matrix_diag

    # Get Cooks Distance contour lines
    x = np.linspace(leverage.min(), leverage.max(), 100)
    params_len = len(lm.params)

    # Calculate lowess for smoothing line
    grid, yhat = lowess(std_resid, leverage).T

    # Draw cooks distance contours, scatter and lowess line
    plt.plot(x, np.sqrt((.5 * params_len * (1 - x)) / x), 'r--')
    plt.plot(x, np.sqrt((1 * params_len * (1 - x)) / x), 'r--')
    plt.plot(x, np.negative(np.sqrt((.5 * params_len * (1 - x)) / x)), 'r--')
    plt.plot(x, np.negative(np.sqrt((1 * params_len * (1 - x)) / x)), 'r--')
    plt.plot(grid, yhat, 'r-')
    plt.plot(leverage,
             std_resid,
             'o',
             mec=edge_col,
             markeredgewidth=1,
             fillstyle='none')

    # Limit y axis to actual values (otherwise contour lines disturb scale)
    plt.ylim(std_resid.min() * 1.1, std_resid.max() * 1.1)

    # Draw Annotations
    for point in top_3:
        plt.annotate(point, xy=(leverage[point], std_resid[point]), color='r')

    # Set Labels
    plt.title('Residuals vs. Leverage', fontsize=title_size)
    plt.xlabel('Leverage')
    plt.ylabel('Standardized residuals')

    return plt


def plot(lm):
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
