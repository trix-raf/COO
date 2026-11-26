from pathlib import Path
from typing import Any
import numpy as np


def format_high_precision_report(
    problem_code: str, title: str, beta_true: np.ndarray, result: dict[str, Any]
) -> str:
    """Menyusun laporan presisi 17-digit signifikan (float64) beserta jarak ULP."""
    est = result["best_params"]
    abs_err = np.abs(est - beta_true)
    rel_err = abs_err / np.maximum(np.abs(beta_true), np.finfo(np.float64).tiny)
    ulp_err = abs_err / np.spacing(beta_true)

    lines = [
        "=" * 92,
        f"[{problem_code}] {title}",
        "-" * 92,
        f"{'Param':<7} | {'Beta Sejati (17-digit)':<25} | {'Estimasi Akhir (17-digit)':<25} | {'Bobot':<7}",
        "-" * 92,
    ]
    for j in range(len(beta_true)):
        lines.append(
            f"beta_{j+1:<2} | {beta_true[j]:+.17e} | {est[j]:+.17e} | {result['learned_weights'][j]:7.3f}"
        )

    lines.extend([
        "-" * 92,
        f"{'Param':<7} | {'Galat Absolut (Sci)':<25} | {'Galat Absolut (Desimal)':<22} | {'Galat Relatif':<12} | {'ULP'}",
        "-" * 92,
    ])
    for j in range(len(beta_true)):
        ulp_str = f"{ulp_err[j]:.1f} ULP" if ulp_err[j] < 1e6 else f"{ulp_err[j]:.2e}"
        lines.append(
            f"beta_{j+1:<2} | {abs_err[j]:.17e} | {abs_err[j]:.18f} | {rel_err[j]:.4e} | {ulp_str}"
        )

    lines.extend([
        "-" * 92,
        f"Loss Akhir    : {result['best_loss']:.17e}",
        f"Switch Orde-2 : Iter-{result['switch_iter']} | Stage-3 Freeze-Polish: Iter-{result.get('stage3_iter')} | Early Stop: Iter-{result['stop_iter']}",
        "=" * 92,
    ])

    return "\n".join(lines)


def emit_report(
    problem_code: str,
    title: str,
    beta_true: np.ndarray,
    result: dict[str, Any],
    save_path: Path | None = None,
) -> None:
    report_text = format_high_precision_report(problem_code, title, beta_true, result)
    print(report_text)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(report_text + "\n", encoding="utf-8")