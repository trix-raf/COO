from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class ProblemSpec:
    """Kontrak standar seluruh masalah uji optimasi nonlinier."""
    code: str
    title: str
    bounds: list[tuple[float, float]]
    beta_true: np.ndarray
    loss_fn: Callable[[np.ndarray], float]
    grad_fn: Callable[[np.ndarray], np.ndarray]
    labels: list[str]
    ncols: int = 2
    x_domain: np.ndarray | None = None
    eval_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None