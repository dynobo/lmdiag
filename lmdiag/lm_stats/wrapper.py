import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass
class LM:
    """Wrapper for linear models which do not include training data/statistics.

    Attributes:
        model: Linear model object, e.g. `sklearn.linear_model.LinearRegression`.
        x: 1-D numpy array with predictor of training data.
        y: 1-D numpy array with response of training data.
    """

    model: Any
    X: np.ndarray
    y: np.ndarray
