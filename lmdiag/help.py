"""Module for printing descriptions to aid interpretation of the different plots."""

import textwrap
from collections.abc import Iterable
from typing import Literal, Union

plot_descriptions = {
    "resid_fit": {
        "Name": "Residuals vs. Fitted",
        "Method": "lmdiag.resid_fit(lm)",
        "x-Axis": (
            "Fitted Values (Predicted y-hat from training data. If a link function "
            "exists, its inverse is applied)"
        ),
        "y-Axis": (
            "Residuals (The 'error' of the model; "
            "Distances of the y-hat values to the fitted regression line)"
        ),
        "Description": (
            "It's purpose is to identify non-linear patterns in the residuals. "
            "If you see a horizontal red line and the points spread around it without "
            "a recognizable pattern, chances are good, that there is no non-linear "
            "relationship in the data. "
            "If you can see clear pattern or a curve, a linear model might not be the "
            "best choice."
            "The red labels show the indices of three observations with the highest "
            "absolute residuals."
        ),
    },
    "q_q": {
        "Name": "Normal Q-Q",
        "Method": "lmdiag.q_q(lm)",
        "x-Axis": "Theoretical Quantiles (Quantiles from the Normal Distribution)",
        "y-Axis": (
            "Standardized residuals (Quantiles of the values of "
            "the dependent variable in sorted order)"
        ),
        "Description": (
            "It's purpose is to check, if the residuals are following a normal "
            "distribution. "
            "It's good, if the points are aligned on the dashed line. "
            "If only a few points are off, take a look at the other plots. "
            "If lots of points do not follow the line, your distribution might be off "
            "normal, e.g. regarding skew, tails or modality."
        ),
    },
    "scale_loc": {
        "Name": "Scale-Location",
        "Method": "lm.scale_loc(lm)",
        "x-Axis": (
            "Fitted Values (Predicted y-hat from training data. If a link function "
            "exists, its inverse is applied)"
        ),
        "y-Axis": "Square root of the absolute value of the Standardized Residuals.",
        "Description": (
            "It's purpose is to check 'homoscedasticity', the assumption of equal "
            "variance. "
            "The plot shows, if the residuals are spread equally across the range of "
            "predictors (fitted values). "
            "The red line should be horizontal and the scatter points equally spread "
            "in a random matter. "
            "The red labels are the indices of the observations with the highest "
            "absolute residuals."
        ),
    },
    "resid_lev": {
        "Name": "Residuals vs. Leverage",
        "Method": "lmdiag.resid_lev(lm)",
        "x-Axis": (
            "Leverage (The 'influence' of an observation. "
            "A measure of how far away the dependent variable's value of an "
            "observation is from those of other observations.)"
        ),
        "y-Axis": (
            "Residuals (The 'error' of the model; "
            "Distances of the y values to the fitted regression line)"
        ),
        "dashed-Lines": "Cook's Distance, 0.5 (inner) and 1 (outer).",
        "Description": (
            "It's purpose is to identify observations with high influence on "
            "calculating the regression. "
            "Those observation might but not have to be outliers, they are just "
            "extreme cases concerning the regression. "
            "The pattern of the scatter points is not relevant here: interesting are "
            "observations in the top right and bottom right of the plot. "
            "If we have cases outside the Cook's Distance (dashed lines), "
            "removing those would have an high impact on our regression line. "
            "The red labels are the indices of the most influential observations."
        ),
    },
}


# Note on extensive type hints: for some reason, type aliases are not revealed at least
# in VSCode function popovers. Therefore, the duplication seems useful.
# ONHOLD: Verify if this is still true, especially with LiteralString in 3.11.
def _print_desc(
    method: Literal["resid_fit", "q_q", "scale_loc", "resid_lev", "plot"],
) -> None:
    """Print description of plot nicely formatted."""
    for key, val in plot_descriptions[method].items():
        wrapper = textwrap.TextWrapper(
            initial_indent=f"{key:>12}: ", width=79, subsequent_indent=" " * 14
        )
        print(wrapper.fill(val), end="\n\n")  # noqa: T201


def help(  # noqa: A001 # shadowing built-in
    method: Union[
        Iterable[Literal["resid_fit", "q_q", "scale_loc", "resid_lev", "plot"]],
        Literal["resid_fit", "q_q", "scale_loc", "resid_lev", "plot"],
    ] = "plot",
) -> None:
    """Prints description(s) of specified plot method(s) to aid interpretation.

    Args:
        method: The name(s) of the plot method(s) for which to print descriptions.
            Defaults to `"plot"`, which prints descriptions of all four plots.
    """
    if isinstance(method, str):
        method = (
            ["resid_fit", "q_q", "scale_loc", "resid_lev"]
            if method == "plot"
            else [method]
        )

    for m in method:
        if m in plot_descriptions:
            _print_desc(m)
        else:
            print(  # noqa: T201
                f"Unknown plotting method '{method}'."
                "Run lmdiag.help() to print all available method descriptions."
            )
