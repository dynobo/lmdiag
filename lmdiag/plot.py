"""Module for Diagnosis Plots of Lineare Regression Models."""
# Standard
import math

# Extra Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table, OLSInfluence
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import norm
import linearmodels

# GLOBAL STYLE VARIABLES
# ----------------


title_size = 15
edge_col = (0, 0, 0, 0.6)


# GATHERING VALUES FOR STATSMODEL
# --------------------------------


class statsmodels_values:
    def __init__(self, lm):
        self.lm = lm

    def get_residuals(self):
        _, data, _ = summary_table(self.lm, alpha=0.05)
        residuals = data[:, 8]
        return residuals

    def get_fitted_values(self):
        """Return 1-D numpy array with fitted values."""
        fitted = self.lm.fittedvalues
        # Transform series to 1-d array, if necessary
        if isinstance(fitted, pd.Series):
            fitted = fitted.values
        return fitted

    def get_standard_residuals(self):
        vals = OLSInfluence(self.lm).summary_frame()
        std_resid = vals["standard_resid"].values
        return std_resid

    def get_sqrt_abs_residuals(self):
        """Return sqrt(|Standardized resiudals|)."""
        std_resid = self.get_standard_residuals()
        sqrt_abs_res = np.sqrt(np.abs(std_resid))
        return sqrt_abs_res

    def get_normalized_quantiles(self):
        val_count = len(self.lm.fittedvalues)
        positions = (np.arange(1.0, val_count + 1)) / (val_count + 1.0)
        norm_quantiles = norm.ppf(positions)
        return norm_quantiles

    def get_cooks_d(self):
        vals = OLSInfluence(self.lm).summary_frame()
        cooks_d = vals["cooks_d"].values
        return cooks_d

    def get_leverage(self):
        infl = self.lm.get_influence()
        return infl.hat_matrix_diag


# GATHERING VALUES FOR LINEARMODELS
# ---------------------------------


class linearmodels_values:
    def __init__(self, lm):
        self.lm = lm

    def get_residuals(self):
        return self.lm.resids

    def get_fitted_values(self):
        """Return 1-D numpy array with fitted values."""
        fitted = self.lm.fitted_values

        # Transform series to 1-d array, if necessary
        if isinstance(fitted, pd.core.frame.DataFrame):
            fitted = fitted.values[:, 0]

        return fitted

    def get_standard_residuals(self):
        X = self.lm.model._x[:, 1]
        mean_X = np.mean(X)
        diff_mean_sqr = np.dot((X - mean_X), (X - mean_X))
        residuals = self.get_residuals()
        h_ii = (X - mean_X) ** 2 / diff_mean_sqr + (1 / self.lm.nobs)
        Var_e = math.sqrt(self.lm.resid_ss / (self.lm.nobs - 2))
        SE_regression = Var_e * ((1 - h_ii) ** 0.5)
        std_resid = residuals / SE_regression
        return std_resid

    def get_sqrt_abs_residuals(self):
        """Return sqrt(|Standardized resiudals|)."""
        std_resid = self.get_standard_residuals()
        sqrt_abs_res = np.sqrt(np.abs(std_resid))
        return sqrt_abs_res

    def get_normalized_quantiles(self):
        val_count = len(self.get_fitted_values())
        positions = (np.arange(1.0, val_count + 1)) / (val_count + 1.0)
        norm_quantiles = norm.ppf(positions)
        return norm_quantiles

    def get_cooks_d(self):
        X = self.lm.model._x[:, 1]
        mean_X = np.mean(X)
        diff_mean_sqr = np.dot((X - mean_X), (X - mean_X))
        h_ii = (X - mean_X) ** 2 / diff_mean_sqr + (1 / self.lm.nobs)
        cooks_d2 = self.get_standard_residuals() ** 2 / len(self.lm.params)
        cooks_d2 *= h_ii / (1 - h_ii)
        return cooks_d2

    def get_leverage(self):
        X = self.lm.model._x[:, 1]
        mean_X = np.mean(X)
        diff_mean_sqr = np.dot((X - mean_X), (X - mean_X))
        h_ii = (X - mean_X) ** 2 / diff_mean_sqr + (1 / self.lm.nobs)
        return h_ii


# SELECT VALUE METHODS ACCORDING TO MODEL TYPE
# --------------------------------------------


def select_model_type(lm):
    """Check if input parameter is an linear regression model."""
    if isinstance(lm, sm.regression.linear_model.RegressionResultsWrapper):
        return statsmodels_values(lm)
    elif isinstance(lm, linearmodels.iv.results.OLSResults):
        return linearmodels_values(lm)
    else:
        raise TypeError(
            "Input is no statsmodel OLS model and no linearmodels OLS model!"
        )


# DRAW CHARTS
# ---------------


def resid_fit(lm):
    """Draw Residuals vs. Fitted Values Plot."""
    model_values = select_model_type(lm)

    # Calculate values for scatter points
    fitted = model_values.get_fitted_values()
    residuals = model_values.get_residuals()

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


def q_q(lm):
    """Draw Q-Q-Plot."""
    model_values = select_model_type(lm)

    # Calulate values for scatter points
    std_resid = model_values.get_standard_residuals()
    quantiles = model_values.get_normalized_quantiles()

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


def scale_loc(lm):
    """Draw Scale-Location Plot."""
    model_values = select_model_type(lm)

    # Get Fitted Values
    fitted_vals = model_values.get_fitted_values()
    sqrt_abs_res = model_values.get_sqrt_abs_residuals()

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


def resid_lev(lm):
    """Draw Stanardized Residuals vs. Leverage Plot."""
    model_values = select_model_type(lm)

    # Get stanardized residuals & cooks distance
    std_resid = model_values.get_standard_residuals()
    cooks_d = model_values.get_cooks_d()

    # Get top three observations for annotation
    top_3 = cooks_d.argsort()[-3:][::1]

    # Get Leverage
    leverage = model_values.get_leverage()

    # Get Cooks Distance contour lines
    x = np.linspace(leverage.min(), leverage.max(), 100)
    params_len = len(lm.params)

    # Calculate lowess for smoothing line
    grid, yhat = lowess(std_resid, leverage).T

    # Draw cooks distance contours, scatter and lowess line
    plt.plot(x, np.sqrt((0.5 * params_len * (1 - x)) / x), "r--")
    plt.plot(x, np.sqrt((1.0 * params_len * (1 - x)) / x), "r--")
    plt.plot(x, np.negative(np.sqrt((0.5 * params_len * (1 - x)) / x)), "r--")
    plt.plot(x, np.negative(np.sqrt((1.0 * params_len * (1 - x)) / x)), "r--")
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
