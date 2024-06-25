from abc import ABC, abstractmethod
from functools import cached_property, wraps
from typing import Any, Callable

import numpy as np
from scipy import special


def optionally_cached_property(func: Callable) -> property:
    """Cache property of a StatsBase instance, if caching is enabled."""
    cached = cached_property(func)

    @wraps(func)
    def wrapper(cls: Any) -> Any:
        if getattr(cls, "_use_cache", False):
            if not hasattr(cached, "__wrapped__"):
                cached.__set_name__(cls, func.__name__)
            return cached.__get__(cls)
        return func(cls)

    return property(wrapper)


class StatsBase(ABC):
    _use_cache: bool

    @optionally_cached_property
    @abstractmethod
    def residuals(self) -> np.ndarray:
        """Distance of actual y from predicted y-hat."""
        ...

    @optionally_cached_property
    @abstractmethod
    def fitted_values(self) -> np.ndarray:
        """Estimated y-hat values of training data, after applying inverse link func."""
        ...

    @optionally_cached_property
    @abstractmethod
    def standard_residuals(self) -> np.typing.ArrayLike: ...

    @optionally_cached_property
    @abstractmethod
    def cooks_d(self) -> np.typing.ArrayLike: ...

    @optionally_cached_property
    @abstractmethod
    def leverage(self) -> np.typing.ArrayLike: ...

    @optionally_cached_property
    @abstractmethod
    def parameter_count(self) -> int:
        """Degrees of freedom of the model; Count of variables + intercept."""
        ...

    @optionally_cached_property
    def sqrt_abs_residuals(self) -> np.ndarray:
        """Square root of absolute standardized residuals."""
        return np.sqrt(np.abs(self.standard_residuals))

    @optionally_cached_property
    def normalized_quantiles(self) -> np.ndarray:
        val_count = len(self.fitted_values)
        positions = (np.arange(1.0, val_count + 1)) / (val_count + 1.0)
        # ndtri is used under the hood by ppf, but is much faster. It skips validation
        # and processing the loc and scale arguments, which are not needed here.
        return special.ndtri(positions)
