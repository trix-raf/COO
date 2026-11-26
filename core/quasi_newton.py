"""
================================================================================
MODUL: core/quasi_newton.py
PERAN: Penggabungan Solver Orde-2 (Stage 2) dan Super-Finetuner Sub-Ruang (Stage 3)
================================================================================

ARSITEKTUR KOMPONEN DI DALAM FILE INI:
1. [FUNGSI UTILITAS MATEMATIKA]
   - `build_multisecant_hessian`: Membangun aproksimasi matriks Hessian dari riwayat
     partikel swarm menggunakan Generalisasi Secant (Generalized Broyden + Tikhonov).
   - `solve_residual_root_gauss_newton`: Mencari akar residu r(beta) = 0 memakai SVD
     Levenberg-Marquardt agar sensitivitas parameter mikro O(u) tidak terkuadratkan.

2. [KOMPONEN STAGE 2: ProjectedDampedBFGS]
   - Solver Kuasi-Newton Orde-2 dengan inisialisasi Hessian berbobot atau Multi-Secant,
     proyeksi gradien pada batas kotak (box constraints), Armijo backtracking line search,
     dan auto-reset saat menemui punggungan nonlinier curam.

3. [KOMPONEN STAGE 3: FrozenSubspaceSuperFinetuner]
   - Membekukan parameter dominan agar dimensi komputasi menyusut hanya ke sub-ruang
     parameter lemah/datar, lalu mengeksekusi 4 tahap pemolesan berurutan:
     * Stage-3A: Subspace Differential Evolution (DE/best/1 & DE/rand/1 + LHS).
     * Stage-3B: Subspace Local Refinement (BFGS -> Trust-Region LM -> Nelder-Mead
                 -> SVD Residual-Root Gauss-Newton).
     * Stage-3C: Joint Full-Space Polish (BFGS penuh -> Trust-Region LM penuh
                 -> Full-Space SVD Residual-Root Gauss-Newton).
     * Stage-3D: Global Multi-Start LHS Residual-Root Escape (cadangan bila dimensi >= 10D
                 masih terjebak pada residu di atas toleransi).
================================================================================
"""

from typing import Callable
import numpy as np


# ==============================================================================
# [BAGIAN 1: FUNGSI UTILITAS MATEMATIKA ORDE-2 & JACOBIAN RESIDU]
# ==============================================================================

def build_multisecant_hessian(
    theta_history: np.ndarray,
    grad_history: np.ndarray,
    reg: float = 1e-8,
) -> np.ndarray:
    """
    [METODE: Generalisasi Secant Multi-Dimensi / Generalized Broyden]
    Memanfaatkan pasangan (beta_k, grad_k) dari riwayat evaluasi swarm untuk
    mengonstruksi aproksimasi matriks Hessian H tanpa evaluasi fungsi tambahan.
    """
    dim = theta_history.shape[1]
    if len(theta_history) < 2:
        return np.eye(dim, dtype=np.float64)

    # Susun selisih posisi (dB) dan selisih gradien (dG) terhadap titik terbaik (baris terakhir)
    dB = (theta_history[:-1] - theta_history[-1]).T  # Bentuk: (dim, m)
    dG = (grad_history[:-1] - grad_history[-1]).T    # Bentuk: (dim, m)

    # Selesaikan H * dB ≈ dG memakai regularisasi Tikhonov agar matriks Gram invertible
    gram = dB @ dB.T + reg * np.eye(dim, dtype=np.float64)
    H_sec = (dG @ dB.T) @ np.linalg.inv(gram)
    H_sym = 0.5 * (H_sec + H_sec.T)

    # Proyeksi nilai eigen agar H_sym dijamin positif-definit (arah descent valid)
    eigvals, eigvecs = np.linalg.eigh(H_sym)
    eigvals_clipped = np.clip(np.abs(eigvals), 1e-6, 1e6)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def solve_residual_root_gauss_newton(
    theta_init: np.ndarray,
    bounds: np.ndarray,
    x_domain: np.ndarray,
    y_target: np.ndarray,
    eval_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    max_iters: int = 50,
    active_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    [METODE: SVD Levenberg-Marquardt Residual Root-Finding]
    Menyelesaikan sistem overdetermined r(beta) = eval_fn(beta, x) - y_target = 0.
    Menggunakan dekomposisi SVD langsung pada Jacobian J (N x k) sehingga parameter
    mikro berskala O(u) tidak terkuadratkan menjadi O(u^2) seperti pada matriks J^T J.
    Mendukung mode penuh (active_indices=None) maupun mode sub-ruang terbekukan.
    """
    low, high = bounds[:, 0], bounds[:, 1]
    span = np.maximum(high - low, 1e-12)
    idx = np.arange(len(theta_init)) if active_indices is None else active_indices
    k = len(idx)

    theta = theta_init.copy()
    r = eval_fn(theta, x_domain) - y_target
    loss = float(0.5 * np.mean(r ** 2))
    lam = 1e-7

    # [LOOP UTAMA: Iterasi Gauss-Newton / Levenberg-Marquardt]
    for _ in range(max_iters):
        if loss < 1e-32:
            break

        # 1. Hitung Jacobian parsial/penuh J (N x k) menggunakan beda hingga pusat
        J = np.zeros((len(x_domain), k), dtype=np.float64)
        # [LOOP KOLOM JACOBIAN: Perturbasi tiap parameter aktif]
        for col, j in enumerate(idx):
            eps_j = 1e-7 * span[j]
            t_plus = theta.copy()
            t_minus = theta.copy()
            t_plus[j] = min(high[j], theta[j] + eps_j)
            t_minus[j] = max(low[j], theta[j] - eps_j)
            denom = max(t_plus[j] - t_minus[j], 1e-15)
            J[:, col] = (eval_fn(t_plus, x_domain) - eval_fn(t_minus, x_domain)) / denom

        # 2. Selesaikan sistem teredam [J; sqrt(lam)*D] * step = [-r; 0] via SVD stabil
        col_norms = np.maximum(np.linalg.norm(J, axis=0), 1e-14)
        accepted = False

        # [LOOP TRUST-REGION DAMPING: Penyesuaian faktor redaman lambda]
        for _ in range(18):
            aug_J = np.vstack([J, np.sqrt(lam) * np.diag(col_norms)])
            aug_r = np.concatenate([-r, np.zeros(k, dtype=np.float64)])
            step_sub, _, _, _ = np.linalg.lstsq(aug_J, aug_r, rcond=None)

            cand = theta.copy()
            cand[idx] = np.clip(theta[idx] + step_sub, low[idx], high[idx])

            if np.linalg.norm(cand - theta) > 0.0:
                r_cand = eval_fn(cand, x_domain) - y_target
                loss_cand = float(0.5 * np.mean(r_cand ** 2))
                if loss_cand < loss:
                    theta = cand
                    r = r_cand
                    loss = loss_cand
                    lam = max(1e-16, lam * 0.25)
                    accepted = True
                    break
            lam *= 5.0

        if not accepted:
            break

    return theta, loss


# ==============================================================================
# [BAGIAN 2: KOMPONEN STAGE-2 — PROJECTED DAMPED BFGS SOLVER]
# ==============================================================================

class ProjectedDampedBFGS:
    """
    [KOMPONEN STAGE 2: Solver Kuasi-Newton Orde-2]
    Menggunakan aproksimasi invers Hessian (H_inv) yang dapat diinisialisasi dari:
    - Bobot sensitivitas diagonal (`initialize_hessian`), atau
    - Riwayat populasi swarm (`initialize_from_multisecant`).
    Dilengkapi proyeksi gradien pada batas kotak dan Armijo backtracking line search.
    """

    def __init__(self, dim: int, bounds: np.ndarray):
        self.dim = dim
        self.low = bounds[:, 0]
        self.high = bounds[:, 1]
        self.H_inv = np.eye(dim, dtype=np.float64)

    def initialize_hessian(self, param_weights: np.ndarray) -> None:
        """[METODE: Inisialisasi Diagonal Berbobot] Menyetel H_inv awal dari bobot sensitivitas."""
        self.H_inv = np.diag(param_weights / np.min(param_weights))

    def initialize_from_multisecant(
        self,
        theta_history: np.ndarray,
        grad_history: np.ndarray,
        param_weights: np.ndarray,
    ) -> None:
        """[METODE: Inisialisasi Multi-Secant] Membangun H_inv dari riwayat swarm dengan fallback diagonal."""
        try:
            H_sec = build_multisecant_hessian(theta_history, grad_history)
            self.H_inv = np.linalg.inv(H_sec)
        except np.linalg.LinAlgError:
            self.initialize_hessian(param_weights)

    def _project_gradient(self, theta: np.ndarray, g: np.ndarray) -> np.ndarray:
        """[METODE: Proyeksi Batas Aktif] Meng-nol-kan komponen gradien yang mendorong keluar kotak batas."""
        g_proj = g.copy()
        at_low = (theta <= self.low + 1e-14) & (g > 0.0)
        at_high = (theta >= self.high - 1e-14) & (g < 0.0)
        g_proj[at_low | at_high] = 0.0
        return g_proj

    def _line_search(
        self,
        theta: np.ndarray,
        current_loss: float,
        g: np.ndarray,
        direction: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
    ) -> tuple[np.ndarray, bool]:
        """[METODE: Armijo Backtracking Line Search] Mengecilkan langkah alpha hingga syarat descent terpenuhi."""
        alpha = 1.0
        c_armijo = 1e-4

        # [LOOP LINE SEARCH: Pemotongan langkah alpha *= 0.5]
        while alpha > 1e-14:
            candidate = np.clip(theta + alpha * direction, self.low, self.high)
            actual_step = candidate - theta
            if np.linalg.norm(actual_step) > 0.0:
                cand_loss = loss_fn(candidate)
                if cand_loss <= current_loss + c_armijo * np.dot(g, actual_step):
                    return candidate, True
            alpha *= 0.5

        return theta.copy(), False

    def step(
        self,
        theta: np.ndarray,
        current_loss: float,
        loss_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        param_weights: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """[METODE: Satu Langkah Pembaruan BFGS] Menghitung arah kuasi-Newton dan memperbarui H_inv."""
        g_raw = grad_fn(theta)
        g = self._project_gradient(theta, g_raw)
        if np.linalg.norm(g) < 1e-15:
            return theta, current_loss

        direction = -self.H_inv @ g
        if np.dot(g, direction) >= 0:
            self.initialize_hessian(param_weights)
            direction = -self.H_inv @ g

        x_next, accepted = self._line_search(theta, current_loss, g, direction, loss_fn)
        if not accepted:
            # Auto-reset invers Hessian bila punggungan nonlinier membuat arah BFGS macet
            self.initialize_hessian(param_weights)
            direction = -self.H_inv @ g
            x_next, accepted = self._line_search(theta, current_loss, g, direction, loss_fn)
            if not accepted:
                return theta, current_loss

        g_next = self._project_gradient(x_next, grad_fn(x_next))
        s = x_next - theta
        y = g_next - g
        sy = float(np.dot(s, y))

        # Pembaruan formula invers BFGS standar dengan syarat kurvatur s^T y > 0
        if sy > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
            rho = 1.0 / sy
            eye = np.eye(self.dim)
            v_mat = eye - rho * np.outer(s, y)
            self.H_inv = v_mat @ self.H_inv @ v_mat.T + rho * np.outer(s, s)

        return x_next, float(loss_fn(x_next))


# ==============================================================================
# [BAGIAN 3: KOMPONEN STAGE-3 — FROZEN-SUBSPACE SUPER-FINETUNER]
# ==============================================================================

class FrozenSubspaceSuperFinetuner:
    """
    [KOMPONEN STAGE 3: Super-Finetuner Sub-Ruang Terbekukan]
    Dirancang khusus untuk menembus lembah datar / parameter lemah dengan rasio sinyal mikro:
    - Membekukan parameter dominan (indeks di luar `weak_mask`).
    - Tahap 3A: Eksplorasi global sub-ruang via Differential Evolution (DE/best/1 & DE/rand/1).
    - Tahap 3B: Pemolesan lokal sub-ruang via BFGS -> Trust-Region LM -> Nelder-Mead
                -> SVD Residual-Root Gauss-Newton.
    - Tahap 3C: Joint Full-Space Polish (melepas pembekuan agar kopling antar-parameter
                menyatu hingga batas presisi float64).
    - Tahap 3D: Global Multi-Start LHS Residual-Root Escape untuk ruang dimensi tinggi.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        rng: np.random.Generator,
        de_pop_per_dim: int = 18,
        de_iters: int = 120,
        sub_bfgs_iters: int = 80,
        tr_iters: int = 60,
        nm_iters: int = 150,
        joint_bfgs_iters: int = 80,
        joint_polish_iters: int = 60,
    ):
        self.bounds = bounds
        self.low = bounds[:, 0]
        self.high = bounds[:, 1]
        self.rng = rng
        self.de_pop_per_dim = de_pop_per_dim
        self.de_iters = de_iters
        self.sub_bfgs_iters = sub_bfgs_iters
        self.tr_iters = tr_iters
        self.nm_iters = nm_iters
        self.joint_bfgs_iters = joint_bfgs_iters
        self.joint_polish_iters = joint_polish_iters

    def _assemble_full_theta(
        self, base_theta: np.ndarray, weak_indices: np.ndarray, sub_theta: np.ndarray
    ) -> np.ndarray:
        """[UTILITAS] Menyisipkan vektor sub-ruang aktif kembali ke vektor parameter penuh."""
        full = base_theta.copy()
        full[weak_indices] = sub_theta
        return full

    # --------------------------------------------------------------------------
    # [METODE STAGE-3A: Subspace Differential Evolution (DE/best/1 & DE/rand/1)]
    # --------------------------------------------------------------------------
    def _subspace_differential_evolution(
        self,
        base_theta: np.ndarray,
        weak_indices: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
    ) -> tuple[np.ndarray, float]:
        k = len(weak_indices)
        sub_low = self.low[weak_indices]
        sub_high = self.high[weak_indices]
        pop_size = max(48, self.de_pop_per_dim * k)

        # Inisialisasi populasi terstratifikasi (Latin Hypercube Sampling) pada sub-ruang
        cut = np.linspace(0.0, 1.0, pop_size + 1)
        u = self.rng.uniform(size=(pop_size, k))
        pop = u * (cut[1:, None] - cut[:-1, None]) + cut[:-1, None]
        for j in range(k):
            self.rng.shuffle(pop[:, j])
        pop = sub_low + pop * (sub_high - sub_low)
        pop[0] = base_theta[weak_indices].copy()

        fitness = np.array(
            [loss_fn(self._assemble_full_theta(base_theta, weak_indices, ind)) for ind in pop],
            dtype=np.float64,
        )
        best_idx = int(np.argmin(fitness))
        best_sub = pop[best_idx].copy()
        best_loss = float(fitness[best_idx])

        # [LOOP GENERASI DE: Evolusi populasi pada sub-ruang parameter lemah]
        for _ in range(self.de_iters):
            f_scale = float(self.rng.uniform(0.45, 0.92))
            cr = float(self.rng.uniform(0.75, 0.98))

            # [LOOP INDIVIDU DE: Mutasi hibrida & binomial crossover]
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                r0, r1, r2 = self.rng.choice(idxs, size=3, replace=False)

                if i % 2 == 0:
                    mutant = best_sub + f_scale * (pop[r1] - pop[r2])  # DE/best/1
                else:
                    mutant = pop[r0] + f_scale * (pop[r1] - pop[r2])   # DE/rand/1

                mutant = np.clip(mutant, sub_low, sub_high)
                cross_mask = self.rng.random(k) < cr
                cross_mask[int(self.rng.integers(0, k))] = True
                trial = np.where(cross_mask, mutant, pop[i])

                trial_loss = float(
                    loss_fn(self._assemble_full_theta(base_theta, weak_indices, trial))
                )
                if trial_loss <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_loss
                    if trial_loss < best_loss:
                        best_loss = trial_loss
                        best_sub = trial.copy()

        return best_sub, best_loss

    # --------------------------------------------------------------------------
    # [METODE STAGE-3B.1: Subspace Projected Damped BFGS]
    # --------------------------------------------------------------------------
    def _subspace_bfgs(
        self,
        base_theta: np.ndarray,
        weak_indices: np.ndarray,
        init_sub: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        n_iters: int,
    ) -> tuple[np.ndarray, float]:
        k = len(weak_indices)
        sub_bounds = self.bounds[weak_indices]
        solver = ProjectedDampedBFGS(dim=k, bounds=sub_bounds)
        weights_sub = np.ones(k, dtype=np.float64)

        def sub_loss(s: np.ndarray) -> float:
            return float(loss_fn(self._assemble_full_theta(base_theta, weak_indices, s)))

        def sub_grad(s: np.ndarray) -> np.ndarray:
            return grad_fn(self._assemble_full_theta(base_theta, weak_indices, s))[weak_indices]

        curr_sub = init_sub.copy()
        curr_loss = sub_loss(curr_sub)

        # [LOOP SUBSPACE BFGS]
        for _ in range(n_iters):
            next_sub, next_loss = solver.step(curr_sub, curr_loss, sub_loss, sub_grad, weights_sub)
            if np.linalg.norm(next_sub - curr_sub) < 1e-16:
                break
            curr_sub, curr_loss = next_sub, next_loss

        return curr_sub, curr_loss

    # --------------------------------------------------------------------------
    # [METODE STAGE-3B.2: Subspace Regularized Trust-Region Levenberg-Marquardt]
    # --------------------------------------------------------------------------
    def _subspace_trust_region_lm(
        self,
        base_theta: np.ndarray,
        weak_indices: np.ndarray,
        init_sub: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        n_iters: int,
    ) -> tuple[np.ndarray, float]:
        k = len(weak_indices)
        sub_low = self.low[weak_indices]
        sub_high = self.high[weak_indices]
        sub_span = np.maximum(sub_high - sub_low, 1e-12)

        curr_sub = init_sub.copy()
        curr_loss = float(loss_fn(self._assemble_full_theta(base_theta, weak_indices, curr_sub)))
        damping = 1e-8

        # [LOOP TRUST-REGION LM: Aproksimasi Hessian lokal & pemecahan sistem normal]
        for _ in range(n_iters):
            full_theta = self._assemble_full_theta(base_theta, weak_indices, curr_sub)
            g_sub = grad_fn(full_theta)[weak_indices]
            if np.linalg.norm(g_sub) < 1e-16:
                break

            # Hitung Hessian sub-ruang H_sub (k x k) via beda hingga pusat pada gradien
            H_sub = np.zeros((k, k), dtype=np.float64)
            # [LOOP KOLOM HESSIAN]
            for j in range(k):
                eps_j = 1e-6 * sub_span[j]
                s_plus = curr_sub.copy()
                s_minus = curr_sub.copy()
                s_plus[j] = min(sub_high[j], curr_sub[j] + eps_j)
                s_minus[j] = max(sub_low[j], curr_sub[j] - eps_j)
                denom = max(s_plus[j] - s_minus[j], 1e-15)

                g_plus = grad_fn(self._assemble_full_theta(base_theta, weak_indices, s_plus))[weak_indices]
                g_minus = grad_fn(self._assemble_full_theta(base_theta, weak_indices, s_minus))[weak_indices]
                H_sub[:, j] = (g_plus - g_minus) / denom

            H_sym = 0.5 * (H_sub + H_sub.T)
            HtH = H_sym.T @ H_sym
            Htg = H_sym.T @ g_sub
            diag_scale = np.maximum(np.diag(HtH), 1e-16)

            accepted = False
            # [LOOP INNER DAMPING LM]
            for _ in range(20):
                lhs = HtH + damping * np.diag(diag_scale)
                try:
                    step = -np.linalg.solve(lhs, Htg)
                except np.linalg.LinAlgError:
                    step = -g_sub / (np.linalg.norm(g_sub) + 1e-12)

                cand_sub = np.clip(curr_sub + step, sub_low, sub_high)
                if np.linalg.norm(cand_sub - curr_sub) > 0.0:
                    cand_loss = float(
                        loss_fn(self._assemble_full_theta(base_theta, weak_indices, cand_sub))
                    )
                    if cand_loss < curr_loss:
                        curr_sub = cand_sub
                        curr_loss = cand_loss
                        damping = max(1e-15, damping * 0.3)
                        accepted = True
                        break
                damping *= 4.0

            if not accepted:
                break

        return curr_sub, curr_loss

    # --------------------------------------------------------------------------
    # [METODE STAGE-3B.3: Subspace Nelder-Mead Simplex (Derivative-Free)]
    # --------------------------------------------------------------------------
    def _subspace_nelder_mead(
        self,
        base_theta: np.ndarray,
        weak_indices: np.ndarray,
        init_sub: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
    ) -> tuple[np.ndarray, float]:
        k = len(weak_indices)
        sub_low = self.low[weak_indices]
        sub_high = self.high[weak_indices]
        sub_span = np.maximum(sub_high - sub_low, 1e-12)

        # Bangun simpleks awal berdimensi k+1 titik
        simplex = np.zeros((k + 1, k), dtype=np.float64)
        simplex[0] = init_sub.copy()
        for j in range(k):
            pt = init_sub.copy()
            delta = 0.01 * sub_span[j]
            pt[j] = pt[j] + delta if pt[j] + delta <= sub_high[j] else pt[j] - delta
            simplex[j + 1] = np.clip(pt, sub_low, sub_high)

        f_vals = np.array(
            [loss_fn(self._assemble_full_theta(base_theta, weak_indices, p)) for p in simplex],
            dtype=np.float64,
        )

        # Koefisien adaptif dimensi untuk refleksi, ekspansi, kontraksi, & penyusutan
        alpha = 1.0
        gamma = 1.0 + 2.0 / max(k, 1)
        rho = 0.75 - 1.0 / (2.0 * max(k, 1))
        sigma = 1.0 - 1.0 / max(k, 1)

        # [LOOP NELDER-MEAD SIMPLEX]
        for _ in range(self.nm_iters):
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]

            if np.max(np.abs(simplex[1:] - simplex[0])) < 1e-15:
                break

            centroid = np.mean(simplex[:-1], axis=0)
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), sub_low, sub_high)
            fr = float(loss_fn(self._assemble_full_theta(base_theta, weak_indices, xr)))

            if f_vals[0] <= fr < f_vals[-2]:
                simplex[-1], f_vals[-1] = xr, fr
            elif fr < f_vals[0]:
                xe = np.clip(centroid + gamma * (xr - centroid), sub_low, sub_high)
                fe = float(loss_fn(self._assemble_full_theta(base_theta, weak_indices, xe)))
                if fe < fr:
                    simplex[-1], f_vals[-1] = xe, fe
                else:
                    simplex[-1], f_vals[-1] = xr, fr
            else:
                xc = np.clip(centroid + rho * (simplex[-1] - centroid), sub_low, sub_high)
                fc = float(loss_fn(self._assemble_full_theta(base_theta, weak_indices, xc)))
                if fc < f_vals[-1]:
                    simplex[-1], f_vals[-1] = xc, fc
                else:
                    for i in range(1, k + 1):
                        simplex[i] = np.clip(
                            simplex[0] + sigma * (simplex[i] - simplex[0]), sub_low, sub_high
                        )
                        f_vals[i] = float(
                            loss_fn(self._assemble_full_theta(base_theta, weak_indices, simplex[i]))
                        )

        best_idx = int(np.argmin(f_vals))
        return simplex[best_idx].copy(), float(f_vals[best_idx])

    # --------------------------------------------------------------------------
    # [METODE STAGE-3D: Global Multi-Start LHS Residual-Root Gauss-Newton]
    # --------------------------------------------------------------------------
    def _global_multistart_residual_root(
        self,
        best_theta: np.ndarray,
        best_loss: float,
        x_domain: np.ndarray,
        y_obs: np.ndarray,
        eval_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        n_starts: int = 24,
    ) -> tuple[np.ndarray, float]:
        """
        Cadangan eksplorasi global berdimensi tinggi:
        Menyebar `n_starts` titik awal Latin Hypercube di seluruh ruang pencarian dan
        menarik masing-masing ke akar sistem residu r(beta) = 0 via SVD Gauss-Newton.
        """
        dim = len(best_theta)
        cut = np.linspace(0.0, 1.0, n_starts + 1)
        u = self.rng.uniform(size=(n_starts, dim))
        lhs = u * (cut[1:, None] - cut[:-1, None]) + cut[:-1, None]
        for j in range(dim):
            self.rng.shuffle(lhs[:, j])
        starts = self.low + lhs * (self.high - self.low)

        curr_best = best_theta.copy()
        curr_loss = best_loss

        # [LOOP MULTI-START LHS]
        for s_idx in range(n_starts):
            cand_theta, cand_loss = solve_residual_root_gauss_newton(
                starts[s_idx],
                self.bounds,
                x_domain,
                y_obs,
                eval_fn,
                max_iters=45,
                active_indices=None,
            )
            if cand_loss < curr_loss:
                curr_loss = cand_loss
                curr_best = cand_theta.copy()
                if curr_loss < 1e-30:
                    break

        return curr_best, curr_loss

    # --------------------------------------------------------------------------
    # [METODE STAGE-3E: Variable Projection (VarPro) Nonlinear-Only Gauss-Newton]
    # --------------------------------------------------------------------------
    def _partition_linear_nonlinear(
        self,
        theta: np.ndarray,
        x_domain: np.ndarray,
        eval_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Mendeteksi otomatis parameter mana yang tampil secara linier pada eval_fn(theta, x)
        dengan menguji apakah turunan kedua d^2(eval_fn)/d(theta_j)^2 == 0.
        """
        dim = len(theta)
        span = np.maximum(self.high - self.low, 1e-12)
        linear_idx = []
        nonlin_idx = []

        for j in range(dim):
            eps = 1e-4 * span[j]
            t0 = theta.copy()
            tp = theta.copy()
            tm = theta.copy()
            tp[j] = min(self.high[j], theta[j] + eps)
            tm[j] = max(self.low[j], theta[j] - eps)
            t0[j] = 0.5 * (tp[j] + tm[j])
            h = max(tp[j] - t0[j], 1e-12)

            fp = eval_fn(tp, x_domain)
            f0 = eval_fn(t0, x_domain)
            fm = eval_fn(tm, x_domain)
            second_diff = np.max(np.abs(fp - 2.0 * f0 + fm)) / (h ** 2)

            if second_diff < 1e-7 * (np.max(np.abs(f0)) + 1.0):
                linear_idx.append(j)
            else:
                nonlin_idx.append(j)

        return np.array(linear_idx, dtype=int), np.array(nonlin_idx, dtype=int)

    def _varpro_nonlinear_gauss_newton(
        self,
        best_theta: np.ndarray,
        best_loss: float,
        x_domain: np.ndarray,
        y_obs: np.ndarray,
        eval_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        n_starts: int = 80,
    ) -> tuple[np.ndarray, float]:
        """
        Memproyeksikan parameter linier secara analitik di dalam evaluasi residu:
        1. Menyaring `n_starts` kandidat Latin Hypercube hanya pada sub-ruang nonlinier
           (misal 4D pada P7: beta_6, beta_8, beta_11, beta_13) via lstsq tanpa clip kasar.
        2. Menjalankan Levenberg-Marquardt pada residu terproyeksi r_vp(theta_nl) = 0,
           lalu mengunci seluruh parameter di ruang penuh dengan rcond bertahap.
        """
        lin_idx, nl_idx = self._partition_linear_nonlinear(best_theta, x_domain, eval_fn)
        if len(lin_idx) == 0 or len(nl_idx) == 0:
            return best_theta, best_loss

        n_pts = len(x_domain)
        n_lin = len(lin_idx)
        n_nl = len(nl_idx)
        nl_low = self.low[nl_idx]
        nl_high = self.high[nl_idx]
        nl_span = np.maximum(nl_high - nl_low, 1e-12)

        def solve_linear_given_nl(theta_nl: np.ndarray, rcond: float = 1e-13) -> tuple[np.ndarray, np.ndarray, float]:
            base = best_theta.copy()
            base[nl_idx] = theta_nl
            base[lin_idx] = 0.0
            offset = eval_fn(base, x_domain)

            # Bangun matriks desain basis Phi (N x n_lin)
            Phi = np.zeros((n_pts, n_lin), dtype=np.float64)
            for col, j in enumerate(lin_idx):
                e_vec = base.copy()
                e_vec[j] = 1.0
                Phi[:, col] = eval_fn(e_vec, x_domain) - offset

            c_hat, _, _, _ = np.linalg.lstsq(Phi, y_obs - offset, rcond=rcond)
            full = base.copy()
            full[lin_idx] = c_hat
            res = eval_fn(full, x_domain) - y_obs
            return full, res, float(0.5 * np.mean(res ** 2))

        # 1. Global Latin Hypercube Screening hanya pada dimensi nonlinier (mis. 4D)
        cut = np.linspace(0.0, 1.0, n_starts + 1)
        u = self.rng.uniform(size=(n_starts, n_nl))
        lhs = u * (cut[1:, None] - cut[:-1, None]) + cut[:-1, None]
        for j in range(n_nl):
            self.rng.shuffle(lhs[:, j])
        candidates = nl_low + lhs * nl_span
        candidates[0] = best_theta[nl_idx].copy()

        scored = []
        for cand_nl in candidates:
            _, _, l_val = solve_linear_given_nl(cand_nl, rcond=1e-12)
            scored.append((l_val, cand_nl))
        scored.sort(key=lambda item: item[0])

        curr_best_theta = best_theta.copy()
        curr_best_loss = best_loss

        # 2. Poles 6 kandidat nonlinier terbaik menggunakan VarPro Reduced Gauss-Newton
        for _, init_nl in scored[:6]:
            nl = init_nl.copy()
            full_t, r, loss = solve_linear_given_nl(nl, rcond=1e-13)
            lam = 1e-7

            for _ in range(100):
                if loss < 1e-32:
                    break

                J_nl = np.zeros((n_pts, n_nl), dtype=np.float64)
                for j in range(n_nl):
                    eps_j = max(1e-8 * nl_span[j], 1e-9 * max(abs(nl[j]), 1e-4))
                    nl_p = nl.copy()
                    nl_m = nl.copy()
                    nl_p[j] = min(nl_high[j], nl[j] + eps_j)
                    nl_m[j] = max(nl_low[j], nl[j] - eps_j)
                    denom = max(nl_p[j] - nl_m[j], 1e-16)
                    _, r_p, _ = solve_linear_given_nl(nl_p, rcond=1e-13)
                    _, r_m, _ = solve_linear_given_nl(nl_m, rcond=1e-13)
                    J_nl[:, j] = (r_p - r_m) / denom

                col_norms = np.maximum(np.linalg.norm(J_nl, axis=0), 1e-15)
                accepted = False
                for _ in range(20):
                    aug_J = np.vstack([J_nl, np.sqrt(lam) * np.diag(col_norms)])
                    aug_r = np.concatenate([-r, np.zeros(n_nl, dtype=np.float64)])
                    step_nl, _, _, _ = np.linalg.lstsq(aug_J, aug_r, rcond=1e-14)
                    cand_nl = np.clip(nl + step_nl, nl_low, nl_high)

                    if np.linalg.norm(cand_nl - nl) > 0.0:
                        cand_full, cand_r, cand_loss = solve_linear_given_nl(cand_nl, rcond=1e-13)
                        if cand_loss < loss:
                            nl = cand_nl
                            full_t = cand_full
                            r = cand_r
                            loss = cand_loss
                            lam = max(1e-16, lam * 0.25)
                            accepted = True
                            break
                    lam *= 4.0

                if not accepted:
                    break

            # Kunci presisi akhir dengan satu proyeksi lstsq presisi mesin (rcond=1e-15)
            full_t, _, loss = solve_linear_given_nl(nl, rcond=1e-15)
            within_bounds = np.all((full_t >= self.low - 1e-12) & (full_t <= self.high + 1e-12))
            if within_bounds and loss < curr_best_loss:
                curr_best_theta = np.clip(full_t, self.low, self.high)
                curr_best_loss = loss
                if curr_best_loss < 1e-30:
                    break

        return curr_best_theta, curr_best_loss

    # --------------------------------------------------------------------------
    # [ORKESTRATOR UTAMA STAGE-3]
    # --------------------------------------------------------------------------
    def run(
        self,
        theta: np.ndarray,
        current_loss: float,
        weak_mask: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x_domain: np.ndarray | None = None,
        y_obs: np.ndarray | None = None,
        eval_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> tuple[np.ndarray, float, list[tuple[np.ndarray, float, str]]]:
        """
        Menjalankan rangkaian pemolesan Stage-3 secara berurutan (3A -> 3B -> 3C -> 3D)
        dan mengembalikan jejak (theta, loss, deskripsi_event) untuk pencatatan riwayat.
        """
        trace: list[tuple[np.ndarray, float, str]] = []
        weak_indices = np.where(weak_mask)[0]
        if len(weak_indices) == 0:
            return theta, current_loss, trace

        best_theta = theta.copy()
        best_loss = current_loss
        has_residual_model = (
            x_domain is not None and y_obs is not None and eval_fn is not None
        )

        # === TAHAP 3A: Subspace Differential Evolution (Parameter Dominan Dibekukan) ===
        sub_de, loss_de = self._subspace_differential_evolution(
            best_theta, weak_indices, loss_fn
        )
        if loss_de < best_loss:
            best_theta = self._assemble_full_theta(best_theta, weak_indices, sub_de)
            best_loss = loss_de
        trace.append((
            best_theta.copy(),
            best_loss,
            f"Stage-3A: Frozen-Dominant Subspace DE ({len(weak_indices)}D aktif)",
        ))

        # === TAHAP 3B: Subspace Local Refinement (BFGS + Trust-Region LM + Nelder-Mead + SVD GN) ===
        sub_bfgs, loss_sub_bfgs = self._subspace_bfgs(
            best_theta,
            weak_indices,
            best_theta[weak_indices],
            loss_fn,
            grad_fn,
            n_iters=self.sub_bfgs_iters,
        )
        if loss_sub_bfgs < best_loss:
            best_theta = self._assemble_full_theta(best_theta, weak_indices, sub_bfgs)
            best_loss = loss_sub_bfgs

        sub_lm, loss_lm = self._subspace_trust_region_lm(
            best_theta,
            weak_indices,
            best_theta[weak_indices],
            loss_fn,
            grad_fn,
            n_iters=self.tr_iters,
        )
        if loss_lm < best_loss:
            best_theta = self._assemble_full_theta(best_theta, weak_indices, sub_lm)
            best_loss = loss_lm

        sub_nm, loss_nm = self._subspace_nelder_mead(
            best_theta, weak_indices, best_theta[weak_indices], loss_fn
        )
        if loss_nm < best_loss:
            best_theta = self._assemble_full_theta(best_theta, weak_indices, sub_nm)
            best_loss = loss_nm

        if has_residual_model:
            gn_sub_theta, gn_sub_loss = solve_residual_root_gauss_newton(
                best_theta,
                self.bounds,
                x_domain,
                y_obs,
                eval_fn,
                max_iters=45,
                active_indices=weak_indices,
            )
            if gn_sub_loss < best_loss:
                best_theta = gn_sub_theta
                best_loss = gn_sub_loss

        trace.append((
            best_theta.copy(),
            best_loss,
            "Stage-3B: Subspace BFGS + LM + Nelder-Mead + SVD Residual-Root",
        ))

        # === TAHAP 3C: Joint Full-Space Polish (Semua Parameter Dibuka) ===
        all_indices = np.arange(len(theta))
        joint_bfgs_theta, joint_bfgs_loss = self._subspace_bfgs(
            best_theta,
            all_indices,
            best_theta,
            loss_fn,
            grad_fn,
            n_iters=self.joint_bfgs_iters,
        )
        if joint_bfgs_loss < best_loss:
            best_theta = joint_bfgs_theta
            best_loss = joint_bfgs_loss

        polished_theta, polished_loss = self._subspace_trust_region_lm(
            best_theta,
            all_indices,
            best_theta,
            loss_fn,
            grad_fn,
            n_iters=self.joint_polish_iters,
        )
        if polished_loss < best_loss:
            best_theta = polished_theta
            best_loss = polished_loss

        if has_residual_model:
            gn_full_theta, gn_full_loss = solve_residual_root_gauss_newton(
                best_theta,
                self.bounds,
                x_domain,
                y_obs,
                eval_fn,
                max_iters=60,
                active_indices=None,
            )
            if gn_full_loss < best_loss:
                best_theta = gn_full_theta
                best_loss = gn_full_loss

        trace.append((
            best_theta.copy(),
            best_loss,
            "Stage-3C: Joint Full-Space BFGS + LM + SVD Residual-Root Polish",
        ))

        # === TAHAP 3D: Global Multi-Start Residual-Root Escape (Bila Masih di Atas Target) ===
        if has_residual_model and best_loss > 1e-12:
            ms_theta, ms_loss = self._global_multistart_residual_root(
                best_theta,
                best_loss,
                x_domain,
                y_obs,
                eval_fn,
                n_starts=max(24, 2 * len(theta)),
            )
            if ms_loss < best_loss:
                best_theta = ms_theta
                best_loss = ms_loss
                trace.append((
                    best_theta.copy(),
                    best_loss,
                    "Stage-3D: Multi-Start LHS Residual-Root Gauss-Newton",
                ))


        # === TAHAP 3E: Variable Projection (VarPro) Nonlinear-Only Gauss-Newton ===
        if has_residual_model and best_loss > 1e-14:
            vp_theta, vp_loss = self._varpro_nonlinear_gauss_newton(
                best_theta,
                best_loss,
                x_domain,
                y_obs,
                eval_fn,
                n_starts=max(80, 12 * len(theta)),
            )
            if vp_loss < best_loss:
                best_theta = vp_theta
                best_loss = vp_loss
                trace.append((
                    best_theta.copy(),
                    best_loss,
                    "Stage-3E: VarPro Reduced-Space Nonlinear Gauss-Newton",
                ))
                
        return best_theta, best_loss, trace