"""
================================================================================
MODUL: main.py
PERAN: Entry Point Eksekusi Benchmark & Runner Modular SwarmAdamBFGS
================================================================================
"""

import argparse
from config import AnimationConfig, OptimizerConfig, PathConfig
from core import SwarmAdamBFGS
from problems import get_problem, list_problem_codes
from utils import animate_function_fit, draw_beta_changes, emit_report

# Pengaturan default saat dijalankan langsung (misalnya lewat tombol Run VS Code)
DEFAULT_PROBLEM_CODE = "ALL"  # Pilihan: "P1" s/d "P8", atau "ALL"
SHOW_PLOTS = False            # Ubah ke True jika ingin window matplotlib pop-up


def run_single_problem(
    code: str,
    opt_cfg: OptimizerConfig,
    path_cfg: PathConfig,
    anim_cfg: AnimationConfig,
    show_plots: bool,
) -> None:
    """
    [ALUR EKSEKUSI SATU MASALAH]
    1. Memuat spesifikasi masalah dari auto-registry (`get_problem`).
    2. Menjalankan optimasi 3-Fase (`SwarmAdamBFGS.optimize`).
    3. Mencetak & menyimpan laporan 17-digit (`emit_report`).
    4. Merender animasi 30-detik GIF/MP4 (`animate_function_fit`).
    5. Menyimpan grafik konvergensi statis (`draw_beta_changes`).
    """
    problem = get_problem(code)
    optimizer = SwarmAdamBFGS(bounds=problem.bounds, cfg=opt_cfg)

    y_obs = (
        problem.eval_fn(problem.beta_true, problem.x_domain)
        if (problem.x_domain is not None and problem.eval_fn is not None)
        else None
    )

    result = optimizer.optimize(
        loss_fn=problem.loss_fn,
        grad_fn=problem.grad_fn,
        x_domain=problem.x_domain,
        y_obs=y_obs,
        eval_fn=problem.eval_fn,
    )

    report_path = path_cfg.get_report_path(problem.code)
    grid_path, combined_path = path_cfg.get_plot_paths(problem.code)
    gif_path, mp4_path = path_cfg.get_animation_paths(problem.code)

    emit_report(
        problem_code=problem.code,
        title=problem.title,
        beta_true=problem.beta_true,
        result=result,
        save_path=report_path,
    )

    animate_function_fit(
        problem_code=problem.code,
        title=problem.title,
        beta_true=problem.beta_true,
        result=result,
        x_domain=problem.x_domain,
        eval_fn=problem.eval_fn,
        labels=problem.labels,
        save_path_gif=gif_path if anim_cfg.save_gif else None,
        save_path_mp4=mp4_path if anim_cfg.save_mp4 else None,
        fps=anim_cfg.fps,
        max_frames=anim_cfg.max_frames,
        target_duration_sec=anim_cfg.target_duration_sec,
    )

    draw_beta_changes(
        a=problem.beta_true,
        b=result["beta_history"],
        save_path_grid=grid_path,
        save_path_combined=combined_path,
        labels=problem.labels,
        ncols=problem.ncols,
        show=show_plots,
    )


def main() -> None:
    available_codes = list_problem_codes()
    parser = argparse.ArgumentParser(description="SwarmAdamBFGS Modular Benchmark Runner")
    parser.add_argument(
        "-p",
        "--problem",
        type=str,
        default=DEFAULT_PROBLEM_CODE,
        help=f"Kode masalah ({', '.join(available_codes)}, atau ALL)",
    )
    parser.add_argument("--no-gif", action="store_true", help="Nonaktifkan ekspor GIF")
    parser.add_argument("--no-mp4", action="store_true", help="Nonaktifkan ekspor video MP4")
    parser.add_argument("--show", action="store_true", help="Tampilkan window grafik statis")
    args = parser.parse_args()

    path_cfg = PathConfig()
    opt_cfg = OptimizerConfig()
    anim_cfg = AnimationConfig(
        save_gif=not args.no_gif,
        save_mp4=not args.no_mp4,
    )
    show_plots = SHOW_PLOTS or args.show

    target = args.problem.strip().upper()
    codes = available_codes if target == "ALL" else [target]

    for code in codes:
        run_single_problem(code, opt_cfg, path_cfg, anim_cfg, show_plots)


if __name__ == "__main__":
    main()