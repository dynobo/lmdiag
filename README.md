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

lmdiag generates plots for _fitted_ linear regression models from
[`statsmodels`](https://www.statsmodels.org/stable/index.html) or
[`linearmodels`](https://bashtage.github.io/linearmodels/doc/index.html).

You can find many examples in
[this jupyter notebook](https://github.com/dynobo/lmdiag/blob/master/example.ipynb).

### Example

```python
import numpy as np
import statsmodels.api as sm
import lmdiag

# Fit model with random sample data
np.random.seed(20)
predictor = np.random.normal(size=30, loc=20, scale=3)
response = 5 + 5 * predictor + np.random.normal(size=30)
X = sm.add_constant(predictor)
lm = sm.OLS(response, X).fit()

# Plot lmdiag facet chart
lmdiag.style.use(style="black_and_red")  # Mimic R's plot.lm style
fig = lmdiag.plot(lm)
fig.show()
```

![image](https://raw.githubusercontent.com/dynobo/lmdiag/master/example.png)

### Methods

- Draw matrix of all plots:

  `lmdiag.plot(lm)`

- Draw individual plots:

  `lmdiag.resid_fit(lm)`

  `lmdiag.q_q(lm)`

  `lmdiag.scale_loc(lm)`

  `lmdiag.resid_lev(lm)`

- Print description to aid plot interpretation:

  `lmdiag.info()` (for all plots)

  `lmdiag.info('<method name>')` (for individual plot)

### Performance

Plotting models fitted on large datasets can be slow. There are some things you can try
to speed it up:

#### 1. Tune LOWESS-parameters

The red smoothing lines are calculated using the "Locally Weighted Scatterplot
Smoothing" algorithm, which can be quite expensive. Try a _lower_ value for `lowess_it`
and a _higher_ value for `lowess_delta` to gain speed at the cost of accuracy:

```python
lmdiag.plot(lm, lowess_it=1, lowess_delta=0.02)
# Defaults are: lowess_it=2, lowess_delta=0.005
```

(For details about those parameters, see
[statsmodels docs](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html).)

#### 2. Change matplotlib backend

Try a different
[matplotlib backend](https://matplotlib.org/stable/users/explain/figure/backends.html).
Especially static backends like `AGG` or `Cairo` should be faster, e.g.:

```python
import matplotlib
matplotlib.use('agg')
```

## Development

### Setup environment

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
```

### Certification

![image](https://raw.githubusercontent.com/dynobo/lmdiag/master/badge.png)
