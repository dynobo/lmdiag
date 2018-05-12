lmdiag
-------- 

To use (with caution), simply do:: 

	>>> import lmdiag
	>>> import statsmodels.api as sm
	>>>
	>>> # Build example lineare regression model:
	>>> predictor = sm.add_constant(predictor)
	>>> response =
    >>> lm = sm.OLS(response, predictor).fit()
    >>>
    >>> # Draw diagnosis plots as matrix:
	>>> lmdiag.plot(lm) 