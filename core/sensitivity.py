"""
================================================================================
MODUL: core/sensitivity.py
PERAN: Komponen 1 — Adaptive Hessian-Curvature & Flat-Interval Weighting
================================================================================

KOMPONEN & METODE DI DALAM FILE INI:
1. `compute_weights`: Mengukur kelengkungan diagonal Hessian tak-berdimensi
   S_j = H_jj * span_j^2 + |g_j| * span_j di sekitar gbest dan memetakan rasio
   sensitivitas menjadi bobot eksplorasi w_j yang aktif seiring cosine decay.
2. `detect_insensitive_subspace`: Memindai interval datar (tempat perubahan loss
   relatif < flat_loss_rel_tol) lalu memakai Adaptive Log-Gap Partitioning pada
   log(bobot) untuk memisahkan parameter dominan (dibekukan di Stage-3) dari
   parameter lemah/datar (aktif di Stage-3).
================================================================================
"""

from typing import Callable
import numpy as np


class SensitivityWeighter:
    """
    Komponen 1: Menghitung bobot eksplorasi tiap parameter berbasis kelengkungan
    Hessian diagonal serta mendeteksi sub-ruang parameter lemah/datar untuk Stage-3.
    """

    def __init__(
        self,
        weight_power: float,
        max_weight: float,
        flat_loss_rel_tol: float = 1e-3,
        eps: float = 1e-8,
    ):
        self.weight_power = weight_power
        self.max_weight = max_weight
        self.flat_loss_rel_tol = flat_loss_rel_tol
        self.eps = eps

    def compute_weights(
        self,
        theta: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        decay_progress: float,
    ) -> np.ndarray:
        """
        [METODE: Estimasi Bobot Kelengkungan Hessian Diagonal]
        Mengembalikan vektor bobot w_j >= 1.0 di mana parameter bergradien lemah
        mendapat bobot lebih besar saat decay_progress -> 1.
        """
        low, high = bounds[:, 0], bounds[:, 1]
        bound_span = np.maximum(high - low, self.eps)
        dim = len(theta)

        g_center = grad_fn(theta)
        curvature_diag = np.zeros(dim, dtype=np.float64)

        # [LOOP DIMENSI: Beda hingga pusat untuk turunan kedua diagonal H_jj]
        for j in range(dim):
            delta = 1e-5 * bound_span[j]
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[j] = min(high[j], theta[j] + delta)
            theta_minus[j] = max(low[j], theta[j] - delta)
            actual_span = max(theta_plus[j] - theta_minus[j], self.eps)

            g_plus = grad_fn(theta_plus)[j]
            g_minus = grad_fn(theta_minus)[j]
            curvature_diag[j] = abs(g_plus - g_minus) / actual_span

        sensitivity = curvature_diag * (bound_span ** 2) + np.abs(g_center) * bound_span
        rel_sensitivity = sensitivity / (np.max(sensitivity) + self.eps)

        target_weights = np.clip(
            (rel_sensitivity + 1e-5) ** (-self.weight_power),
            1.0,
            self.max_weight,
        )
        return (1.0 - decay_progress) * 1.0 + decay_progress * target_weights

    def detect_insensitive_subspace(
        self,
        theta: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        bounds: np.ndarray,
        base_weights: np.ndarray,
        weak_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        [METODE: Flat-Interval Scanning + Adaptive Log-Gap Subspace Partitioning]
        1. Mengukur lebar interval datar Delta_flat_j di mana |L(theta + d*e_j) - L_0|
           hampir tidak berubah (< flat_loss_rel_tol * |L_0|).
        2. Menggabungkan bobot kelengkungan dengan rasio kedataran.
        3. Memisahkan kelompok parameter dominan vs parameter lemah menggunakan
           lompatan rasio terbesar pada ruang log(bobot).
        """
        low, high = bounds[:, 0], bounds[:, 1]
        span = np.maximum(high - low, self.eps)
        dim = len(theta)

        curr_weights = np.maximum(
            base_weights,
            self.compute_weights(theta, grad_fn, bounds, decay_progress=1.0),
        )

        base_loss = float(loss_fn(theta))
        loss_tol = max(self.flat_loss_rel_tol * abs(base_loss), 1e-24)
        flat_ratios = np.zeros(dim, dtype=np.float64)

        # [LOOP PEMINDAIAN INTERVAL DATAR TIAP PARAMETER]
        for j in range(dim):
            max_flat_delta = 0.0
            for frac in (1e-4, 1e-3, 1e-2, 5e-2, 1.5e-1, 3e-1):
                step = frac * span[j]
                for sign in (-1.0, 1.0):
                    cand = theta.copy()
                    cand[j] = np.clip(theta[j] + sign * step, low[j], high[j])
                    if abs(float(loss_fn(cand)) - base_loss) <= loss_tol:
                        max_flat_delta = max(max_flat_delta, abs(cand[j] - theta[j]))
            flat_ratios[j] = max_flat_delta / span[j]

        flat_boost = 1.0 + 15.0 * flat_ratios
        combined_weights = np.clip(curr_weights * flat_boost, 1.0, self.max_weight)

        # Pemisahan kelompok dominan vs lemah via Adaptive Log-Gap Partitioning
        if dim >= 2:
            sorted_idx = np.argsort(combined_weights)
            sorted_w = combined_weights[sorted_idx]
            log_gaps = np.diff(np.log(np.maximum(sorted_w, 1e-12)))
            max_gap_pos = int(np.argmax(log_gaps))

            if log_gaps[max_gap_pos] > np.log(2.2):
                adaptive_cut = float(
                    np.sqrt(sorted_w[max_gap_pos] * sorted_w[max_gap_pos + 1])
                )
                weak_mask = combined_weights >= adaptive_cut
            else:
                weak_mask = combined_weights >= weak_threshold
        else:
            weak_mask = combined_weights >= weak_threshold

        # Jaminan minimal 1 parameter aktif bila semua terklasifikasi dominan
        if not np.any(weak_mask):
            weak_mask[int(np.argmax(combined_weights))] = True

        return combined_weights, weak_mask