import timeit

import numpy as np
import statsmodels as sm
import statsmodels.api as sma

import lmdiag
import lmdiag.statistics.select

df = sma.datasets.get_rdataset("ames", "openintro").data
lm = sm.formula.api.ols("np.log10(price) ~ Q('Overall.Qual') + np.log(area)", df).fit()

lm_stats = lmdiag.statistics.select.get_stats(lm)


if __name__ == "__main__":
    import timeit

    for stmt in [
        "lm_stats.residuals",
        "lm_stats.fitted_values",
        "lm_stats.standard_residuals",
        "lm_stats.cooks_d",
        "lm_stats.leverage",
        "lm_stats.parameter_count",
        "lm_stats.sqrt_abs_residuals",
        "lm_stats.normalized_quantiles",
        "lmdiag.plot(lm)",
        "lmdiag.resid_fit(lm)",
        "lmdiag.q_q(lm)",
        "lmdiag.scale_loc(lm)",
        "lmdiag.resid_lev(lm)",
    ]:
        timing = timeit.repeat(
            stmt=stmt,
            globals=globals(),
            number=1,
            repeat=3,
        )
        print(  # noqa: T201
            f"{stmt:<30} "
            f"Max: {np.max(timing):.3f}"
            f"\tMin: {np.min(timing):.3f}"
            f"\tAvg: {np.mean(timing):.3f}"
        )
