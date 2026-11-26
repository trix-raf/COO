"""
================================================================================
MODUL: core/stochastic_ops.py
PERAN: Operator Populasi Swarm (LHS, Boundary Reflection, Re-Heat, Subspace Re-Search)
================================================================================
"""

from typing import Callable
import numpy as np
from core.schedules import clip_gradient


def latin_hypercube_sampling(
    rng: np.random.Generator, n_samples: int, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """
    [METODE: Stratified Latin Hypercube Sampling (LHS)]
    Menyebar partikel secara merata di setiap stratum dimensi [low, high].
    """
    dim = len(low)
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    u = rng.uniform(size=(n_samples, dim))
    points = u * (cut[1:, None] - cut[:-1, None]) + cut[:-1, None]
    for j in range(dim):
        rng.shuffle(points[:, j])
    return low + points * (high - low)


def apply_boundary_reflection(
    pos: np.ndarray,
    vel: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    damping: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """
    [METODE: Damped Boundary Reflection]
    Memantulkan partikel yang menabrak batas kotak dengan meredam komponen kelajuannya.
    """
    hit_low = pos < low
    hit_high = pos > high
    if np.any(hit_low):
        pos = np.where(hit_low, low, pos)
        vel = np.where(hit_low, -damping * vel, vel)
    if np.any(hit_high):
        pos = np.where(hit_high, high, pos)
        vel = np.where(hit_high, -damping * vel, vel)
    return pos, vel


def select_diverse_basins(
    pbest: np.ndarray,
    pbest_loss: np.ndarray,
    gbest: np.ndarray,
    bound_span: np.ndarray,
    k: int = 6,
) -> list[np.ndarray]:
    """
    [METODE: Greedy Spatial Diversity Basin Selection]
    Memilih hingga `k` titik kandidat terbaik dari `pbest` yang terpisah jauh secara
    spasial untuk diadu pada turnamen Multi-Basin BFGS saat transisi ke Fase 2.
    """
    order = np.argsort(pbest_loss)
    selected: list[np.ndarray] = [gbest.copy()]
    min_dist = 0.08  # Jarak ternormalisasi minimal antar cekungan

    # [LOOP KANDIDAT PBEST]
    for idx in order:
        if len(selected) >= k:
            break
        cand = pbest[idx]
        dists = [np.linalg.norm((cand - s) / bound_span) for s in selected]
        if min(dists) >= min_dist:
            selected.append(cand.copy())

    return selected


def stochastic_reheat(
    rng: np.random.Generator,
    X: np.ndarray,
    V: np.ndarray,
    M: np.ndarray,
    S: np.ndarray,
    gbest: np.ndarray,
    bounds: np.ndarray,
    param_weights: np.ndarray,
    decay_ratio: float,
) -> None:
    """
    [METODE: Hybrid Global-Local Stochastic Re-Heat]
    Separuh partikel melakukan eksplorasi lokal di sekitar `gbest` (diskalakan oleh
    `param_weights`), sementara separuh lainnya memindai ulang ruang secara global (LHS).
    """
    low, high = bounds[:, 0], bounds[:, 1]
    span = high - low
    n_particles, dim = X.shape
    half = n_particles // 2

    weight_scale = np.sqrt(param_weights / np.min(param_weights))
    local_scale = (0.04 + 0.16 * decay_ratio) * span * weight_scale

    X[:half] = np.clip(
        gbest + rng.normal(0.0, local_scale, size=(half, dim)), low, high
    )
    X[half:] = latin_hypercube_sampling(rng, n_particles - half, low, high)
    X[0] = gbest.copy()

    V.fill(0.0)
    M.fill(0.0)
    S.fill(0.0)


def stratified_subspace_research(
    rng: np.random.Generator,
    n_particles: int,
    gbest: np.ndarray,
    bounds: np.ndarray,
    weak_mask: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    relax_steps: int = 30,
    relax_lr_scale: float = 0.08,
    clip_norm: float = 50.0,
) -> np.ndarray:
    """
    [METODE: Targeted Weak-Subspace Stratified Re-Search]
    Menahan parameter dominan di sekitar `gbest` dan menyebar ulang partikel secara
    terstratifikasi khusus pada dimensi lemah (`weak_mask`), diikuti relaksasi Adam singkat.
    """
    low, high = bounds[:, 0], bounds[:, 1]
    span = high - low
    dim = len(gbest)

    X_new = np.tile(gbest, (n_particles, 1))
    strong_mask = ~weak_mask
    if np.any(strong_mask):
        jitter = rng.normal(0.0, 1e-4 * span[strong_mask], size=(n_particles, int(np.sum(strong_mask))))
        X_new[:, strong_mask] = np.clip(
            gbest[strong_mask] + jitter, low[strong_mask], high[strong_mask]
        )

    weak_indices = np.where(weak_mask)[0]
    if len(weak_indices) > 0:
        sub_lhs = latin_hypercube_sampling(
            rng, n_particles, low[weak_indices], high[weak_indices]
        )
        X_new[:, weak_indices] = sub_lhs

    X_new[0] = gbest.copy()

    # Relaksasi Adam singkat khusus pada dimensi lemah
    m = np.zeros_like(X_new)
    s = np.zeros_like(X_new)
    lr_sub = relax_lr_scale * span
    for step in range(1, relax_steps + 1):
        for i in range(1, n_particles):
            g = clip_gradient(grad_fn(X_new[i]), max_norm=clip_norm)
            g_sub = np.where(weak_mask, g, 0.0)
            m[i] = 0.9 * m[i] + 0.1 * g_sub
            s[i] = 0.999 * s[i] + 0.001 * (g_sub ** 2)
            m_hat = m[i] / (1.0 - 0.9 ** step)
            s_hat = s[i] / (1.0 - 0.999 ** step)
            X_new[i] = np.clip(
                X_new[i] - lr_sub * m_hat / (np.sqrt(s_hat) + 1e-8), low, high
            )

    return X_new