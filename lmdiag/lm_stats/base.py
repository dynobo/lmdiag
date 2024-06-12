from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm


class StatsBase(ABC):
    @property
    @abstractmethod
    def residuals(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def fitted_values(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def standard_residuals(self) -> np.typing.ArrayLike: ...

    @property
    @abstractmethod
    def cooks_d(self) -> np.typing.ArrayLike: ...

    @property
    @abstractmethod
    def leverage(self) -> np.typing.ArrayLike: ...

    @property
    @abstractmethod
    def params_count(self) -> int: ...

    @property
    def sqrt_abs_residuals(self) -> np.ndarray:
        return np.sqrt(np.abs(self.standard_residuals))

    @property
    def normalized_quantiles(self) -> np.ndarray:
        val_count = len(self.fitted_values)
        positions = (np.arange(1.0, val_count + 1)) / (val_count + 1.0)
        return norm.ppf(positions)
