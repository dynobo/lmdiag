# lmdiag

**Python Library providing Diagnostic Plots for Linear Regression Models.** (Like
[plot.lm](https://www.rdocumentation.org/packages/stats/versions/3.5.0/topics/plot.lm)
in R.)

I built this, because I missed the diagnostics plots of R for a university project.
There are some substitutions in Python for individual charts, but they are spread over
different libraries and sometimes don't show the exact same. My implementation tries to
copycat the R-plots, but I didn't reimplement the R-code: The charts are just based on
available documentation.

## Installation

`pip install lmdiag`

## Usage

The plots need a
[fitted Linear Regression Model](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.fit.html)
created by `statsmodels` as input.

Fitted Model from
[`linearmodels`](https://bashtage.github.io/linearmodels/doc/index.html) should also
work, however, that's not tested very well.

### Example

(See also the more extensive
[Example Notebook](https://github.com/dynobo/lmdiag/blob/master/example.ipynb))

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

![image](https://raw.githubusercontent.com/dynobo/lmdiag/master/example.png)

### Methods

- Draw matrix of all plots:

  `lmdiag.plot(lm)`

- Draw individual plots:

  `lmdiag.resid_fit(lm)`

  `lmdiag.q_q(lm)`

  `lmdiag.scale_loc(lm)`

  `lmdiag.resid_lev(lm)`

- Print useful descriptions for interpretations:

  `lmdiag.info()` (for all plots)

  `lmdiag.info('<method name>')` (for individual plot)

## Development

### Setup environment

```sh
python -m venv .venv
source .venv/bin/activate
pip install '.[dev]'
pre-commit install
```

### Certification

![image](https://raw.githubusercontent.com/dynobo/lmdiag/master/badge.png)
