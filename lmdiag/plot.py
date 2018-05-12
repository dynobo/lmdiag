"""Module for Diagnosis Plots of Lineare Regression Models."""

# Standard Libs
import textwrap

# Extra Libs
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table, OLSInfluence
from statsmodels.nonparametric.smoothers_lowess import lowess

# Global Style Variables
title_size = 15
edge_col = (0, 0, 0, 0.6)

# Plot descriptions as help for interpretation
desc = {
        'resid_fit': {
            'Name': 'Residuals vs. Fitted',
            'Method': 'lmdiag.resid_fit(lm)',
            'x-Axis': ('Residuals (The "error" of the model; '
                       'Distance to the fitted regression line)'),
            'y-Axis': ('Fitted Values (The dependent variable of your model; '
                       'What you threw in statsmodels OLS as 1st parameter)'),
            'Description': ('test. '
                            'The red labels show the indices of three '
                            'observations with the highest residuals.')
            },
        'q_q': {
            'Name': 'Normal Q-Q',
            'Method': 'lmdiag.q_q(lm)',
            'x-Axis': ('Theoretical Quantiles (TODO '
                       ')'),
            'y-Axis': ('Standardized residuals (TODO '
                       ')'),
            'Description': 'Bla bla bla'
            }
        }


def verify_input(lm):
    """Check if input parameter is an linear regression model."""
    if not isinstance(lm, sm.regression.linear_model.RegressionResultsWrapper):
        raise TypeError('Input is no statsmodel OLS model!')
    return


def resid_fit(lm):
    """Draw Residuals vs. Fitted Values Plot."""
    verify_input(lm)

    # Calculate values for scatter points
    tbl, data, labels = summary_table(lm, alpha=0.05)
    fitted = lm.fittedvalues
    residuals = data[:, 8]

    # Calculate lowess for smoothing line
    grid, yhat = lowess(residuals, fitted).T

    # Get top three observations for annotation
    top_3 = residuals.argsort()[-3:][::1]

    # Draw scatter and lowess line
    plt.plot([fitted.min(), fitted.max()], [0, 0], 'k:')
    plt.plot(grid, yhat, 'r-')
    plt.plot(fitted, residuals, 'o', mec=edge_col, fillstyle='none')

    # Draw Annotations
    for point in top_3:
        plt.annotate(point,
                     xy=(fitted.values[point], residuals[point]),
                     color='r')

    # Set Labels
    plt.title('Residual vs. Fitted', fontsize=title_size)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')

    ax = plt.gca()
    return ax


def q_q(lm):
    """Draw Q-Q-Plot."""
    verify_input(lm)

    # Calulate values for scatter points
    vals = OLSInfluence(lm).summary_frame()
    std_resid = vals['standard_resid'].sort_values().values
    quantiles = np.random.normal(0, 1, len(std_resid))
    quantiles = np.sort(quantiles)

    # Function for fitted line
    fit = np.polyfit(quantiles, std_resid, deg=1)

    # Get top three observations for annotation
    # (need position of sorted for coord, and original for label)
    top_3_sorted = np.abs(std_resid).argsort()[-3:][::1]
    top_3_orig = np.abs(vals['standard_resid'].values).argsort()[-3:][::1]
    top_3 = zip(top_3_sorted, top_3_orig)

    # Draw scatter and fitted line
    plt.plot(quantiles, fit[0] * quantiles + fit[1], 'r:')
    plt.plot(quantiles, std_resid, 'o', mec=edge_col, fillstyle='none')

    # Draw Annotations
    for point in top_3:
        plt.annotate(point[1],
                     xy=(quantiles[point[0]], std_resid[point[0]]),
                     color='r')

    # Set Labels
    plt.title('Normal Q-Q', fontsize=title_size)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Standardized residuals')

    ax = plt.gca()
    return ax


def scale_loc(lm):
    """Draw Scale-Location Plot."""
    verify_input(lm)

    # Get Fitted Values
    tbl, data, labels = summary_table(lm, alpha=0.05)
    fitted_vals = lm.fittedvalues

    # Get sqrt(|Standardized resiudals|)
    vals = OLSInfluence(lm).summary_frame()
    std_resid = vals['standard_resid'].values
    sqrt_abs_res = np.sqrt(np.abs(std_resid))

    # Get top three observations for annotation
    top_3 = sqrt_abs_res.argsort()[-3:][::1]

    # Calculate lowess for smoothing line
    grid, yhat = lowess(sqrt_abs_res, fitted_vals).T

    # Draw scatter and lowess line
    plt.plot(grid, yhat, 'r-')
    plt.plot(fitted_vals, sqrt_abs_res, 'o', mec=edge_col, fillstyle='none')

    # Draw Annotations
    for point in top_3:
        plt.annotate(point,
                     xy=(fitted_vals.values[point], sqrt_abs_res[point]),
                     color='r')

    # Set Labels
    plt.title('Scale-Location', fontsize=title_size)
    plt.xlabel('Fitted values')
    plt.ylabel(r'$\sqrt{\left|Standardized\ residuals\right|}$')

    ax = plt.gca()
    return ax


def resid_lev(lm):
    """Draw Stanardized Residuals vs. Leverage Plot."""
    verify_input(lm)

    # Get stanardized residuals & cooks distance
    vals = OLSInfluence(lm).summary_frame()
    std_resid = vals['standard_resid'].values
    cooks_d = vals['cooks_d'].values

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
    plt.plot(leverage, std_resid, 'o', mec=edge_col, fillstyle='none')

    # Limit y axis to actual values (otherwise contour lines disturb scale)
    plt.ylim(std_resid.min() * 1.1, std_resid.max() * 1.1)

    # Draw Annotations
    for point in top_3:
        plt.annotate(point, xy=(leverage[point], std_resid[point]), color='r')

    # Set Labels
    plt.title('Residuals vs. Leverage', fontsize=title_size)
    plt.xlabel('Leverage')
    plt.ylabel('Standardized residuals')

    ax = plt.gca()
    return ax


def print_desc(plotname):
    """Prints description of plot nicely formatted."""
    for key, val in desc[plotname].items():
        wrapper = textwrap.TextWrapper(initial_indent=f'{key:>12}: ',
                                       width=79,
                                       subsequent_indent=' '*14)
        print(wrapper.fill(val))


def info(*args):
    """Prints the description of the plots which names are passed
    as string-argument. If no argument is passed, all descriptions
    are printed"""
    # If no argument, print all descriptions
    if len(args) < 1:
        for d in desc:
            print_desc(d)
            print()
        return
    # Else try to print the description
    for arg in args:
        if (arg in desc.keys()):
            print_desc(arg)
            print()
        else:
            print('Unknown plot. Run lmdiag.info() for all available plots')


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

    ax = plt.gca()
    return ax
