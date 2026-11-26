"""
================================================================================
MODUL: core/early_stopping.py
PERAN: Komponen 2 — Weight-Gated Historical Memory Early Stopping & Stagnation Detector
================================================================================
"""

from collections import deque
import numpy as np


class WeightedMemoryEarlyStopper:
    """
    Komponen 2: Mengelola memori geser (sliding window) untuk gbest dan loss.
    Membedakan dua kondisi penting:
    1. `is_stagnated_above_target`: BFGS macet pada dataran datar sementara loss
       masih di atas `stage3_loss_trigger` -> segera hentikan Fase 2 untuk masuk Stage-3.
    2. `should_stop`: Seluruh dimensi memenuhi `param_weights * norm_std < tol` dan
       loss telah mencapai presisi mesin / konvergensi sejati.
    """

    def __init__(
        self,
        window: int,
        tol: float,
        machine_loss_tol: float,
        stage3_loss_trigger: float,
        min_decay: float,
        min_second_order_steps: int,
        bound_span: np.ndarray,
    ):
        self.window = window
        self.tol = tol
        self.machine_loss_tol = machine_loss_tol
        self.stage3_loss_trigger = stage3_loss_trigger
        self.min_decay = min_decay
        self.dim = len(bound_span)
        self.min_second_order_steps = max(min_second_order_steps, 3 * self.dim)
        self.bound_span = bound_span

        self.gbest_memory: deque[np.ndarray] = deque(maxlen=window)
        self.loss_memory: deque[float] = deque(maxlen=window)

    def update(self, gbest: np.ndarray, loss: float) -> None:
        """Menyimpan titik terbaik dan nilai loss terbaru ke buffer historis."""
        self.gbest_memory.append(gbest.copy())
        self.loss_memory.append(float(loss))

    def is_stagnated_above_target(
        self, step: int, mode: str, switch_iter: int | None, best_loss: float
    ) -> bool:
        """
        [METODE: Deteksi Stagnasi Dataran Datar]
        Mengembalikan True apabila solver Fase 2 (BFGS) berhenti bergerak selama
        `window` iterasi sementara nilai loss masih lebih besar dari `stage3_loss_trigger`.
        """
        if mode != "SECOND_ORDER" or switch_iter is None:
            return False
        if (step - switch_iter) < self.min_second_order_steps or len(self.loss_memory) < self.window:
            return False
        if best_loss <= self.stage3_loss_trigger:
            return False

        loss_arr = np.asarray(self.loss_memory)
        rel_loss_change = abs(loss_arr[0] - loss_arr[-1]) / (abs(loss_arr[0]) + 1e-15)
        return bool(rel_loss_change < 1e-11)

    def should_stop(
        self,
        step: int,
        mode: str,
        switch_iter: int | None,
        decay_progress: float,
        best_loss: float,
        search_std_vec: np.ndarray,
        param_weights: np.ndarray,
    ) -> bool:
        """
        [METODE: Weight-Gated Convergence Check]
        Memastikan parameter berbobot tinggi (lemah/sensitif rendah) ikut stabil
        sebelum menghentikan proses optimasi.
        """
        if len(self.gbest_memory) < self.window:
            return False

        if mode == "SECOND_ORDER":
            if switch_iter is None or (step - switch_iter) < self.min_second_order_steps:
                return False
            if best_loss > self.stage3_loss_trigger:
                return False
            if best_loss <= self.machine_loss_tol:
                mem_arr = np.asarray(self.gbest_memory)
                norm_mem_std = np.std(mem_arr, axis=0) / self.bound_span
                if np.all(param_weights * norm_mem_std < self.tol):
                    return True

        if decay_progress < self.min_decay:
            return False

        mem_arr = np.asarray(self.gbest_memory)
        norm_mem_std = np.std(mem_arr, axis=0) / self.bound_span
        norm_search_std = search_std_vec / self.bound_span

        weighted_mem_score = param_weights * norm_mem_std
        weighted_search_score = param_weights * norm_search_std

        loss_arr = np.asarray(self.loss_memory)
        rel_loss_std = np.std(loss_arr) / (np.abs(np.mean(loss_arr)) + 1e-15)

        return bool(
            np.all(weighted_mem_score < self.tol)
            and np.all(weighted_search_score < self.tol)
            and (rel_loss_std < self.tol or np.std(loss_arr) < 1e-28)
        )