from abc import ABC, abstractmethod
from functools import cached_property, wraps
from typing import Any, Callable

import numpy as np
from scipy.stats import norm


def optionally_cached_property(func: Callable) -> property:
    cached = cached_property(func)

    @wraps(func)
    def wrapper(cls: Any) -> Any:
        if getattr(cls, "_cache_properties", False):
            if not hasattr(cached, "__wrapped__"):
                cached.__set_name__(cls, func.__name__)
            return cached.__get__(cls)
        return func(cls)

    return property(wrapper)


class StatsBase(ABC):
    @optionally_cached_property
    @abstractmethod
    def residuals(self) -> np.ndarray: ...

    @optionally_cached_property
    @abstractmethod
    def fitted_values(self) -> np.ndarray: ...

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
    def params_count(self) -> int: ...

    @optionally_cached_property
    def sqrt_abs_residuals(self) -> np.ndarray:
        return np.sqrt(np.abs(self.standard_residuals))

    @optionally_cached_property
    def normalized_quantiles(self) -> np.ndarray:
        val_count = len(self.fitted_values)
        positions = (np.arange(1.0, val_count + 1)) / (val_count + 1.0)
        return norm.ppf(positions)
