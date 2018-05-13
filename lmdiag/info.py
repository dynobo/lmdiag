"""Prints Plot Descriptions to help the interpretation of the charts."""

# Standard Libs
import textwrap

# Plot descriptions
desc = {
        'resid_fit': {
            'Name': 'Residuals vs. Fitted',
            'Method': 'lmdiag.resid_fit(lm)',
            'x-Axis': ('Fitted Values (The dependent variable of your model; '
                       'What you threw in statsmodels OLS as 1st parameter)'),
            'y-Axis': ('Residuals (The "error" of the model; '
                       'Distance to the fitted regression line)'),
            'Description': ('It\'s purpose is to identify non-linear patterns '
                            'in the residuals. If you see a horizontal red '
                            'line and the points spread around it without a '
                            'recognizable pattern, chances are good, that '
                            'there is no non-linear relationship in the data. '
                            'If you can see clear pattern or a curve, a '
                            'linear model might not be the best choice.'
                            'The red labels show the indices of three '
                            'observations with the highest absolute '
                            'residuals.')
            },
        'q_q': {
            'Name': 'Normal Q-Q',
            'Method': 'lmdiag.q_q(lm)',
            'x-Axis': ('Theoretical Quantiles (Quantiles from the Normal '
                       'Distribution)'),
            'y-Axis': ('Standardized residuals (Quantiles of the values of '
                       'hte dependent variable in sorted order)'),
            'Description': ('It\'s purpose is to check, if the residuals are '
                            'following a normal distribution. It\'s good, '
                            'if the points are aligned on the dashed line. If '
                            'only a few points are off, take a look at the '
                            'other plots. If lot\'s of points do not follow '
                            'the line, your distribution might be off '
                            'normal, e.g. regarding skew, tails or modality.')
            },
        'scale_loc': {
            'Name': 'Scale-Location',
            'Method': 'lm.scale_loc(lm)',
            'x-Axis': ('Fitted Values (The dependent variable of your model; '
                       'What you threw in statsmodels OLS as 1st parameter)'),
            'y-Axis': ('Squareroot of the absolute value of the Standardized '
                       'Residuals.'),
            'Description': ('It\'s purpose is to check "homoscedasticity" '
                            'the assumption of equal variance. '
                            'The plot shows, if the residuals are spread '
                            'equally accross the range of predictors (Fitted '
                            'values). The red line should be horizonzal and '
                            'the scatter points equally spread in a random '
                            'matter. The red labels are the indices of the '
                            'observations with the highest absolute '
                            'residuals.')
            },
        'resid_lev': {
            'Name': 'Residuals vs. Leverage',
            'Method': 'lmdiag.resid_lev(lm)',
            'x-Axis': ('Leverage (The "influence" of an observation. A '
                       'measure of how far away the dependend variables value '
                       'of an observation is from those of other '
                       'observations.)'),
            'y-Axis': ('Residuals (The "error" of the model; '
                       'Distance to the fitted regression line)'),
            'dashed-Lines': 'Cook\' Distance, 0.5 (inner) and 1 (outer).',
            'Description': ('It\'s purpose is to identify observations with '
                            'high influence on calculating the regression. '
                            'Those oberservation might but not have to be '
                            'outliers, they are just extreme cases concerning '
                            'the regression. The pattern of the scatter '
                            'points is not relevant here: interesting are '
                            'observations in the top right and bottom right '
                            'of the plot. If we have cases outside the '
                            'Cook\'s Distance (dashed lines), removing those '
                            'would have an high impact on our regression '
                            'line. The red labels are the indices of the '
                            'most influencal observations.')
            }
        }


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
