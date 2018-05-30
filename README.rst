lmdiag
=======

**Python Library providing Diagnostic Plots for Lineare Regression Models.** (Like `plot.lm <https://www.rdocumentation.org/packages/stats/versions/3.5.0/topics/plot.lm>`_ in R.)

I built this, because I missed the diagnostics plots of R for a university project. There are some substitutions in Python for individual charts, but they are spread over different libraries and sometimes don't show the exact same. My implementation tries to copycat the R-plots, but I didn't reimplement the R-code: The charts are just based on available documentation.

Installation
------------

Available in PyPi: https://pypi.org/project/lmdiag/

- Using pip: ``pip install lmdiag``
- Using pipenv: ``pipenv install lmdiag``

Usage
-----------

The plots need a `fitted Linear Regression Model <https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.fit.html>`_ created by statsmodels as input.

Example
........
(See also the more extensive `Example Notebook <https://github.com/dynobo/lmdiag/blob/master/example.ipynb>`_)

::

        import numpy as np
        import matplotlib.pyplot as plt
        import statsmodels.api as sm
        import lmdiag

        %matplotlib inline  # In Jupyter

        # Generate sample model
        np.random.seed(20)
        predictor = np.random.normal(size=30, loc=20, scale=3)
        response = 5 + 5 * predictor + np.random.normal(size=30)
        X = sm.add_constant(predictor)
        lm = sm.OLS(response, X).fit()

        # Plot chart matrix (and enlarge figure)
        plt.figure(figsize=(10,7))
        lmdiag.plot(lm);


.. image:: https://raw.githubusercontent.com/dynobo/lmdiag/master/example.png


Methods
........

- Draw matrix of all plots:

  ``lmdiag.plot(lm)``

- Draw individual plots:

  ``lmdiag.resid_fit(lm)``

  ``lmdiag.q_q(lm)``

  ``lmdiag.scale_loc(lm)``

  ``lmdiag.resid_lev(lm)``

- Print useful descriptions for interpretations:

  ``lmdiag.info()`` (for all plots)

  ``lmdiag.info('<method name>')`` (for individual plot)

Development
------------

Disclaimer
..........

This is my very first public python library. Don't expect everything to work smoothly. I'm happy to receive useful feedback or pull requests.

Certification
..............
.. image:: https://raw.githubusercontent.com/dynobo/lmdiag/master/badge.png

Packaging and Upload to PyPi
............................

- ``pipenv run rstcheck README.rst`` (check syntax)
- ``rm -rf ./dist`` (delete old builds)
- ``python setup.py sdist``
- ``python setup.py bdist_wheel``
- ``twine upload dist/*``
- Then new release on github...
