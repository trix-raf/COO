import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def draw_beta_changes(
    a,
    b,
    save_path_grid: str | Path | None = "beta_changes_grid.png",
    save_path_combined: str | Path | None = "beta_changes_combined.png",
    labels: list[str] | None = None,
    ncols: int = 2,
    show: bool = True,
) -> None:
    """Membuat 2 window visualisasi konvergensi parameter beta (grid & gabungan)."""
    if hasattr(a, "detach"):
        a_arr = a.detach().cpu().numpy().flatten()
    else:
        a_arr = np.array(
            [float(x.detach().cpu()) if hasattr(x, "detach") else float(x) for x in a],
            dtype=float,
        )

    if hasattr(b, "detach"):
        b_arr = b.detach().cpu().numpy()
    else:
        b_arr = np.array(
            [
                [float(val.detach().cpu()) if hasattr(val, "detach") else float(val) for val in row]
                for row in b
            ],
            dtype=float,
        )

    if b_arr.ndim != 2:
        raise ValueError(f"Input b harus berupa array/list 2D, bukan {b_arr.ndim}D.")

    n_beta = len(a_arr)
    if b_arr.shape[1] == n_beta:
        pass
    elif b_arr.shape[0] == n_beta:
        b_arr = b_arr.T
    else:
        raise ValueError(f"Dimensi b {b_arr.shape} tidak cocok dengan panjang a ({n_beta}).")

    n_iter = b_arr.shape[0]
    iterasi = np.arange(1, n_iter + 1)

    if labels is None:
        labels = [f"$\\beta_{{{i + 1}}}$" for i in range(n_beta)]

    cmap = plt.get_cmap("tab10" if n_beta <= 10 else "tab20")

    # Window 1: Grid Subplot Individual
    ncols_eff = min(ncols, n_beta)
    nrows = math.ceil(n_beta / ncols_eff)
    fig_grid, axes = plt.subplots(
        nrows, ncols_eff, figsize=(5.5 * ncols_eff, 3.8 * nrows), dpi=130, squeeze=False
    )
    fig_grid.canvas.manager.set_window_title("Window 1 - Konvergensi Per Parameter Beta")

    for idx in range(n_beta):
        row, col = divmod(idx, ncols_eff)
        ax = axes[row, col]
        target = a_arr[idx]
        history = b_arr[:, idx]

        ax.plot(
            iterasi,
            history,
            color="#1f77b4",
            linewidth=1.8,
            label=f"Tebakan (Akhir: {history[-1]:.5f})",
        )
        ax.axhline(
            y=target,
            color="#d62728",
            linestyle="--",
            linewidth=1.6,
            label=f"Asli ({target:.5f})",
        )
        ax.scatter(iterasi[0], history[0], color="#ff7f0e", s=30, zorder=5)
        ax.scatter(iterasi[-1], history[-1], color="#2ca02c", s=30, zorder=5)

        ax.set_title(f"Konvergensi {labels[idx]}")
        ax.set_xlabel("Iterasi")
        ax.set_ylabel("Nilai")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8.5, framealpha=0.9)

    for idx in range(n_beta, nrows * ncols_eff):
        row, col = divmod(idx, ncols_eff)
        axes[row, col].axis("off")

    fig_grid.tight_layout()
    if save_path_grid:
        Path(save_path_grid).parent.mkdir(parents=True, exist_ok=True)
        fig_grid.savefig(save_path_grid, bbox_inches="tight")

    # Window 2: Grafik Gabungan
    fig_comb, ax_comb = plt.subplots(figsize=(10, 5.5), dpi=130)
    fig_comb.canvas.manager.set_window_title("Window 2 - Gabungan Konvergensi Semua Beta")

    for idx in range(n_beta):
        color = cmap(idx % cmap.N)
        target = a_arr[idx]
        history = b_arr[:, idx]

        ax_comb.plot(
            iterasi,
            history,
            color=color,
            linewidth=2.0,
            label=f"{labels[idx]} | Akhir: {history[-1]:.5f} (Asli: {target:.5f})",
        )
        ax_comb.axhline(y=target, color=color, linestyle="--", linewidth=1.4, alpha=0.8)
        ax_comb.scatter(iterasi[0], history[0], color=color, s=25, zorder=5)
        ax_comb.scatter(iterasi[-1], history[-1], color=color, s=35, marker="D", zorder=5)

    beta_min = min(float(np.min(b_arr)), float(np.min(a_arr)))
    beta_max = max(float(np.max(b_arr)), float(np.max(a_arr)))
    rentang = beta_max - beta_min
    margin = 0.08 * rentang if rentang > 0 else (abs(beta_max) * 0.1 or 0.1)

    ax_comb.set_xlim(1, n_iter)
    ax_comb.set_ylim(beta_min - margin, beta_max + margin)
    ax_comb.set_title("Gabungan Konvergensi Semua Parameter $\\beta$")
    ax_comb.set_xlabel("Iterasi (Domain)")
    ax_comb.set_ylabel("Nilai $\\beta$ (Kodomain)")
    ax_comb.grid(True, linestyle=":", alpha=0.6)

    ax_comb.plot([], [], color="gray", linestyle="-", linewidth=2.0, label="Kurva Tebakan")
    ax_comb.plot([], [], color="gray", linestyle="--", linewidth=1.4, label="Garis Horizontal Asli")
    ax_comb.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5, framealpha=0.95)

    fig_comb.tight_layout()
    if save_path_combined:
        Path(save_path_combined).parent.mkdir(parents=True, exist_ok=True)
        fig_comb.savefig(save_path_combined, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig_grid)
        plt.close(fig_comb)