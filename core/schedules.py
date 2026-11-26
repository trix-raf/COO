"""
================================================================================
MODUL: core/schedules.py
PERAN: Penjadwalan Parameter Dinamis (Cosine Decay) & Gradient Norm Clipping
================================================================================
"""

import numpy as np


def cosine_decay(step: int, max_steps: int, start: float, end: float) -> float:
    """
    [METODE: Cosine Annealing Schedule]
    Menurunkan nilai parameter secara mulus dari `start` ke `end` berbasis kurva kosinus.
    """
    progress = min(step, max_steps) / max(max_steps, 1)
    cos_term = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(end + (start - end) * cos_term)


def clip_gradient(
    grad: np.ndarray, max_norm: float = 50.0, eps: float = 1e-8
) -> np.ndarray:
    """
    [METODE: L2 Gradient Norm Clipping]
    Membatasi norma L2 gradien agar tidak meledak pada suku eksponensial curam.
    """
    norm = float(np.linalg.norm(grad))
    if norm > max_norm:
        return grad * (max_norm / (norm + eps))
    return grad