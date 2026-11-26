"""
================================================================================
MODUL: core/optimizer.py
PERAN: Orkestrator Utama SwarmAdamBFGS (3-Stage Hybrid Global-Local Optimizer)
================================================================================

ARSITEKTUR KOMPONEN & ALUR EKSEKUSI DI DALAM FILE INI:
1. [INISIALISASI POPULASI & MEMORI]
   - Menskalakan jumlah partikel secara adaptif terhadap dimensi masalah
     (`max(n_particles, min_particles_per_dim * dim)`).
   - Menginisialisasi posisi swarm memakai Latin Hypercube Sampling (LHS),
     momen Adam (M, S), kelajuan inersia (V), serta buffer riwayat Multi-Secant.

2. [FASE 1 — ORDE-1: Swarm-Adam + Cosine Decay + Adaptive Sensitivity Weighting]
   - [LOOP PARTIKEL]: Menghitung gradien terklip, langkah adaptif Adam (m_hat, s_hat),
     tarikan kognitif (pbest), tarikan sosial (gbest), serta pantulan batas teredam.
   - Menghitung bobot sensitivitas parameter berbasis kelengkungan diagonal Hessian
     dan interval datar (`SensitivityWeighter`).
   - Memantau sebaran spasial ternormalisasi RMS (`norm_std_scalar`):
     * Jika ruang menyempit tetapi dimensi lemah terdeteksi -> `stratified_subspace_research`.
     * Jika terjebak pada norma gradien besar -> Hybrid Global-Local `stochastic_reheat`.
     * Jika perubahan gradien kecil atau peluruhan mencapai `force_switch_decay`
       -> Memicu transisi ke Fase 2 (Multi-Basin BFGS Tournament).

3. [FASE 2 — ORDE-2: Multi-Basin Multi-Secant BFGS Tournament]
   - Memilih hingga K cekungan paling beragam dari `pbest` (`select_diverse_basins`),
     menginisialisasi Hessian dari bobot sensitivitas & riwayat Multi-Secant,
     melakukan pemanasan `multi_basin_warmup_steps` langkah BFGS, lalu mengunci
     kandidat pemenang untuk pemolesan kuasi-Newton hingga presisi mesin.

4. [FASE 3 — STAGE-3: Frozen-Dominant Subspace Super-Finetuning]
   - Dipicu otomatis jika BFGS stagnan di dataran datar atau setelah loop utama
     selesai namun `best_loss > stage3_loss_trigger`.
   - Memisahkan indeks parameter lemah via Adaptive Log-Gap Partitioning dan
     menjalankan `FrozenSubspaceSuperFinetuner` (Stage 3A -> 3B -> 3C -> 3D).
================================================================================
"""

from typing import Any, Callable
import numpy as np

from config import OptimizerConfig
from core.early_stopping import WeightedMemoryEarlyStopper
from core.quasi_newton import FrozenSubspaceSuperFinetuner, ProjectedDampedBFGS
from core.schedules import clip_gradient, cosine_decay
from core.sensitivity import SensitivityWeighter
from core.stochastic_ops import (
    apply_boundary_reflection,
    latin_hypercube_sampling,
    select_diverse_basins,
    stochastic_reheat,
    stratified_subspace_research,
)


# ==============================================================================
# [KOMPONEN UTAMA: ORKESTRATOR OPTIMIZER HIBRIDA 3-FASE]
# ==============================================================================

class SwarmAdamBFGS:
    """
    Orkestrator utama yang menggabungkan:
    - Fase 1: Swarm Velocity + Adam Preconditioning + Cosine Decay + Sensitivity Weighting
    - Fase 2: Multi-Basin Multi-Secant Projected Damped BFGS
    - Fase 3: Frozen-Dominant Subspace Super-Finetuning (DE + LM + Nelder-Mead + SVD GN)
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        cfg: OptimizerConfig | None = None,
    ):
        self.cfg = cfg or OptimizerConfig()
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.rng = np.random.default_rng(self.cfg.stage1.seed)

        # Inisialisasi komponen pembobotan sensitivitas (Komponen 1)
        sens_cfg = self.cfg.sensitivity
        self.weighter = SensitivityWeighter(
            weight_power=sens_cfg.weight_power,
            max_weight=sens_cfg.max_param_weight,
            flat_loss_rel_tol=sens_cfg.flat_loss_rel_tol,
            eps=self.cfg.stage1.eps,
        )

        # Inisialisasi solver Orde-2 (Stage 2) dan Super-Finetuner (Stage 3)
        self.bfgs = ProjectedDampedBFGS(dim=self.dim, bounds=self.bounds)
        s3_cfg = self.cfg.stage3
        self.finetuner = FrozenSubspaceSuperFinetuner(
            bounds=self.bounds,
            rng=self.rng,
            de_pop_per_dim=s3_cfg.de_pop_per_dim,
            de_iters=s3_cfg.de_iters,
            sub_bfgs_iters=s3_cfg.sub_bfgs_iters,
            tr_iters=s3_cfg.tr_iters,
            nm_iters=s3_cfg.nm_iters,
            joint_bfgs_iters=s3_cfg.joint_bfgs_iters,
            joint_polish_iters=s3_cfg.joint_polish_iters,
        )

    # --------------------------------------------------------------------------
    # [METODE INTERNAL: Turnamen Multi-Basin BFGS saat Transisi Fase 1 -> Fase 2]
    # --------------------------------------------------------------------------
    def _run_multibasin_bfgs_tournament(
        self,
        pbest: np.ndarray,
        pbest_loss: np.ndarray,
        gbest: np.ndarray,
        gbest_loss: float,
        bound_span: np.ndarray,
        param_weights: np.ndarray,
        secant_thetas: list[np.ndarray],
        secant_grads: list[np.ndarray],
        loss_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, float]:
        """
        Memilih K kandidat cekungan paling beragam dari pbest, memoles masing-masing
        selama `multi_basin_warmup_steps` langkah BFGS (dengan inisialisasi diagonal
        maupun Multi-Secant), lalu mengunci solver pada pemenang dengan loss terendah.
        """
        s2_cfg = self.cfg.stage2
        basins = select_diverse_basins(
            pbest=pbest,
            pbest_loss=pbest_loss,
            gbest=gbest,
            bound_span=bound_span,
            k=s2_cfg.multi_basin_k,
        )

        winner_theta = gbest.copy()
        winner_loss = gbest_loss
        winner_H_inv: np.ndarray | None = None

        has_secant = len(secant_thetas) >= max(4, self.dim + 1)
        theta_hist_arr = np.asarray(secant_thetas, dtype=np.float64) if has_secant else np.empty((0, self.dim))
        grad_hist_arr = np.asarray(secant_grads, dtype=np.float64) if has_secant else np.empty((0, self.dim))

        # [LOOP TURNAMEN CEKUNGAN: Evaluasi tiap kandidat basin dengan BFGS pemanasan]
        for idx, cand_init in enumerate(basins):
            trial_solver = ProjectedDampedBFGS(dim=self.dim, bounds=self.bounds)
            if idx == 0 and has_secant:
                trial_solver.initialize_from_multisecant(
                    theta_hist_arr, grad_hist_arr, param_weights
                )
            else:
                trial_solver.initialize_hessian(param_weights)

            cand_theta = cand_init.copy()
            cand_loss = float(loss_fn(cand_theta))

            # [LOOP PEMANASAN BFGS]
            for _ in range(s2_cfg.multi_basin_warmup_steps):
                next_theta, next_loss = trial_solver.step(
                    cand_theta, cand_loss, loss_fn, grad_fn, param_weights
                )
                if np.linalg.norm(next_theta - cand_theta) < 1e-16:
                    break
                cand_theta, cand_loss = next_theta, next_loss

            if cand_loss < winner_loss:
                winner_loss = cand_loss
                winner_theta = cand_theta.copy()
                winner_H_inv = trial_solver.H_inv.copy()

        if winner_H_inv is not None:
            self.bfgs.H_inv = winner_H_inv
        else:
            self.bfgs.initialize_hessian(param_weights)

        return winner_theta, winner_loss

    # --------------------------------------------------------------------------
    # [METODE UTAMA: EKSEKUSI OPTIMASI 3-FASE]
    # --------------------------------------------------------------------------
    def optimize(
        self,
        loss_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x_domain: np.ndarray | None = None,
        y_obs: np.ndarray | None = None,
        eval_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> dict[str, Any]:
        s1 = self.cfg.stage1
        sens = self.cfg.sensitivity
        s3 = self.cfg.stage3
        stop_cfg = self.cfg.stopping

        low, high = self.bounds[:, 0], self.bounds[:, 1]
        bound_span = np.maximum(high - low, s1.eps)
        n_particles = max(s1.n_particles, s1.min_particles_per_dim * self.dim)

        # === TAHAP 0: Inisialisasi Swarm (LHS), Momen Adam, & Riwayat ===
        X = latin_hypercube_sampling(self.rng, n_particles, low, high)
        V = np.zeros_like(X)
        M = np.zeros_like(X)
        S = np.zeros_like(X)

        pbest = X.copy()
        pbest_loss = np.array([loss_fn(x) for x in X], dtype=np.float64)
        gbest_idx = int(np.argmin(pbest_loss))
        gbest = pbest[gbest_idx].copy()
        gbest_loss = float(pbest_loss[gbest_idx])

        prev_grad = clip_gradient(grad_fn(gbest), s1.grad_clip_norm, s1.eps)
        param_weights = np.ones(self.dim, dtype=np.float64)

        # Buffer pasangan (theta, grad) untuk aproksimasi Multi-Secant Hessian
        max_secant_pairs = max(16, 3 * self.dim)
        secant_thetas: list[np.ndarray] = [gbest.copy()]
        secant_grads: list[np.ndarray] = [grad_fn(gbest)]

        subspace_researched = False
        adam_t = 0
        mode = "FIRST_ORDER"
        switch_iter: int | None = None
        stage3_iter: int | None = None
        stop_iter: int | None = None
        early_stopped = False

        stopper = WeightedMemoryEarlyStopper(
            window=stop_cfg.memory_window,
            tol=stop_cfg.early_stop_tol,
            machine_loss_tol=stop_cfg.machine_loss_tol,
            stage3_loss_trigger=s3.stage3_loss_trigger,
            min_decay=stop_cfg.stop_min_decay,
            min_second_order_steps=stop_cfg.min_second_order_steps,
            bound_span=bound_span,
        )
        stopper.update(gbest, gbest_loss)

        # Pencatatan metrik per iterasi untuk visualisasi, laporan, & animasi
        beta_history = [gbest.copy()]
        std_history = [float(np.sqrt(np.mean((np.std(X, axis=0) / bound_span) ** 2)))]
        weight_history = [param_weights.copy()]
        loss_history = [gbest_loss]
        lr_history = [s1.lr_max]
        event_history = ["Init: Latin Hypercube Swarm"]

        # [LOOP ITERASI UTAMA: Berjalan dari t = 1 hingga max_iter]
        for t in range(1, s1.max_iter + 1):
            lr_t = cosine_decay(t, s1.max_iter, s1.lr_max, s1.lr_min)
            w_t = cosine_decay(t, s1.max_iter, s1.w_max, s1.w_min)
            decay_progress = 1.0 - (lr_t - s1.lr_min) / (s1.lr_max - s1.lr_min + s1.eps)
            event_label = f"{mode} (decay={decay_progress:.0%})"

            # ==================================================================
            # [FASE 1: ORDE-1 — SWARM VELOCITY + ADAM + SENSITIVITY WEIGHTING]
            # ==================================================================
            if mode == "FIRST_ORDER":
                adam_t += 1
                base_lr = max(lr_t, 0.12 * s1.lr_max) if subspace_researched else lr_t
                # Skala dimensi agar langkah Adam konsisten pada ruang berdimensi tinggi
                dim_scale = 1.0 / np.sqrt(max(1.0, self.dim / 4.0))
                eff_lr = np.minimum(base_lr * dim_scale * param_weights, s1.lr_max)

                # [LOOP PARTIKEL SWARM]
                for i in range(n_particles):
                    raw_g = grad_fn(X[i])
                    g = clip_gradient(raw_g, s1.grad_clip_norm, s1.eps)

                    M[i] = s1.adam_beta1 * M[i] + (1.0 - s1.adam_beta1) * g
                    S[i] = s1.adam_beta2 * S[i] + (1.0 - s1.adam_beta2) * (g ** 2)
                    m_hat = M[i] / (1.0 - s1.adam_beta1 ** adam_t)
                    s_hat = S[i] / (1.0 - s1.adam_beta2 ** adam_t)
                    adam_step = m_hat / (np.sqrt(s_hat) + s1.eps)

                    r1 = self.rng.random(self.dim)
                    r2 = self.rng.random(self.dim)
                    V[i] = (
                        w_t * V[i]
                        - eff_lr * adam_step
                        + s1.c1 * r1 * (pbest[i] - X[i])
                        + s1.c2 * r2 * (gbest - X[i])
                    )
                    X[i] = X[i] + V[i]
                    X[i], V[i] = apply_boundary_reflection(
                        X[i], V[i], low, high, s1.boundary_damping
                    )

                    curr_loss = float(loss_fn(X[i]))
                    if curr_loss < pbest_loss[i]:
                        pbest_loss[i] = curr_loss
                        pbest[i] = X[i].copy()
                        if curr_loss < gbest_loss:
                            gbest_loss = curr_loss
                            gbest = X[i].copy()
                            secant_thetas.append(gbest.copy())
                            secant_grads.append(raw_g.copy())
                            if len(secant_thetas) > max_secant_pairs:
                                secant_thetas.pop(0)
                                secant_grads.pop(0)

                # Perbarui bobot sensitivitas berbasis kelengkungan Hessian diagonal
                param_weights = self.weighter.compute_weights(
                    gbest, grad_fn, self.bounds, decay_progress
                )

                # Evaluasi metrik sebaran spasial RMS & perubahan gradien
                search_std_vec = np.std(X, axis=0)
                norm_std_scalar = float(np.sqrt(np.mean((search_std_vec / bound_span) ** 2)))

                curr_grad = clip_gradient(grad_fn(gbest), s1.grad_clip_norm, s1.eps)
                grad_norm = float(np.linalg.norm(curr_grad))
                grad_delta = float(
                    np.linalg.norm(curr_grad - prev_grad) / (grad_norm + s1.eps)
                )
                prev_grad = curr_grad.copy()

                # Pengambilan keputusan stokastik saat populasi mulai mengumpul
                if norm_std_scalar < s1.std_tol:
                    weak_mask = param_weights > sens.weak_weight_threshold
                    if not subspace_researched and decay_progress >= 0.35 and np.any(weak_mask):
                        subspace_researched = True
                        adam_t = 0
                        X = stratified_subspace_research(
                            rng=self.rng,
                            n_particles=n_particles,
                            gbest=gbest,
                            bounds=self.bounds,
                            weak_mask=weak_mask,
                            grad_fn=grad_fn,
                            relax_steps=sens.subspace_relax_steps,
                            relax_lr_scale=sens.subspace_relax_lr_scale,
                            clip_norm=s1.grad_clip_norm,
                        )
                        V.fill(0.0)
                        M.fill(0.0)
                        S.fill(0.0)
                        pbest = X.copy()
                        pbest_loss = np.array([loss_fn(x) for x in X], dtype=np.float64)
                        best_sub = int(np.argmin(pbest_loss))
                        if pbest_loss[best_sub] < gbest_loss:
                            gbest_loss = float(pbest_loss[best_sub])
                            gbest = pbest[best_sub].copy()
                        event_label = "Stage-1: Weak-Subspace Stratified Re-Search"

                    elif grad_norm > s1.grad_norm_tol and decay_progress < s1.force_switch_decay:
                        if self.rng.random() < s1.reheat_prob:
                            stochastic_reheat(
                                self.rng,
                                X,
                                V,
                                M,
                                S,
                                gbest,
                                self.bounds,
                                param_weights,
                                lr_t / s1.lr_max,
                            )
                            event_label = "Stage-1: Hybrid Global-Local Stochastic Re-Heat"

                    elif grad_delta < s1.grad_delta_tol or decay_progress >= s1.force_switch_decay:
                        mode = "SECOND_ORDER"
                        switch_iter = t
                        gbest, gbest_loss = self._run_multibasin_bfgs_tournament(
                            pbest=pbest,
                            pbest_loss=pbest_loss,
                            gbest=gbest,
                            gbest_loss=gbest_loss,
                            bound_span=bound_span,
                            param_weights=param_weights,
                            secant_thetas=secant_thetas,
                            secant_grads=secant_grads,
                            loss_fn=loss_fn,
                            grad_fn=grad_fn,
                        )
                        event_label = "Switch -> Stage-2 Multi-Basin Multi-Secant BFGS"

                elif decay_progress >= s1.force_switch_decay:
                    mode = "SECOND_ORDER"
                    switch_iter = t
                    gbest, gbest_loss = self._run_multibasin_bfgs_tournament(
                        pbest=pbest,
                        pbest_loss=pbest_loss,
                        gbest=gbest,
                        gbest_loss=gbest_loss,
                        bound_span=bound_span,
                        param_weights=param_weights,
                        secant_thetas=secant_thetas,
                        secant_grads=secant_grads,
                        loss_fn=loss_fn,
                        grad_fn=grad_fn,
                    )
                    event_label = "Switch -> Stage-2 Multi-Basin Multi-Secant BFGS"

            # ==================================================================
            # [FASE 2: ORDE-2 — PROJECTED DAMPED BFGS LOCAL POLISH]
            # ==================================================================
            else:
                gbest, gbest_loss = self.bfgs.step(
                    gbest, gbest_loss, loss_fn, grad_fn, param_weights
                )
                search_std_vec = np.zeros(self.dim, dtype=np.float64)
                norm_std_scalar = 0.0

            stopper.update(gbest, gbest_loss)
            beta_history.append(gbest.copy())
            std_history.append(norm_std_scalar)
            weight_history.append(param_weights.copy())
            loss_history.append(gbest_loss)
            lr_history.append(lr_t)
            event_history.append(event_label)

            # Deteksi stagnasi di atas target -> segera putus BFGS untuk masuk Stage-3
            if (
                s3.enable_stage3
                and stopper.is_stagnated_above_target(t, mode, switch_iter, gbest_loss)
            ):
                break

            # Cek kriteria konvergensi presisi mesin / Weight-Gated Early Stopping
            if stopper.should_stop(
                t,
                mode,
                switch_iter,
                decay_progress,
                gbest_loss,
                search_std_vec,
                param_weights,
            ):
                early_stopped = True
                stop_iter = t
                break

        # ======================================================================
        # [FASE 3: STAGE-3 — FROZEN-DOMINANT SUBSPACE SUPER-FINETUNING]
        # ======================================================================
        if s3.enable_stage3 and gbest_loss > s3.stage3_loss_trigger:
            stage3_iter = len(beta_history) - 1
            param_weights, weak_mask = self.weighter.detect_insensitive_subspace(
                theta=gbest,
                loss_fn=loss_fn,
                grad_fn=grad_fn,
                bounds=self.bounds,
                base_weights=param_weights,
                weak_threshold=sens.weak_weight_threshold,
            )
            gbest, gbest_loss, s3_trace = self.finetuner.run(
                theta=gbest,
                current_loss=gbest_loss,
                weak_mask=weak_mask,
                loss_fn=loss_fn,
                grad_fn=grad_fn,
                x_domain=x_domain,
                y_obs=y_obs,
                eval_fn=eval_fn,
            )

            # Rekam setiap kemajuan sub-tahap Stage-3 ke riwayat visualisasi & animasi
            for step_theta, step_loss, step_event in s3_trace:
                beta_history.append(step_theta.copy())
                std_history.append(0.0)
                weight_history.append(param_weights.copy())
                loss_history.append(step_loss)
                lr_history.append(s1.lr_min)
                event_history.append(step_event)

            if gbest_loss <= stop_cfg.machine_loss_tol:
                early_stopped = True
                stop_iter = len(beta_history) - 1

        return {
            "best_params": gbest,
            "best_loss": gbest_loss,
            "switch_iter": switch_iter,
            "stage3_iter": stage3_iter,
            "stop_iter": stop_iter,
            "early_stopped": early_stopped,
            "learned_weights": param_weights,
            "beta_history": np.array(beta_history),
            "std_history": np.array(std_history),
            "weight_history": np.array(weight_history),
            "loss_history": np.array(loss_history),
            "lr_history": np.array(lr_history),
            "event_history": event_history,
        }