"""Module for Diagnosis Plots of Lineare Regression Models."""

# Standard Libs
import textwrap

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
