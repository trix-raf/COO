from __future__ import annotations                      # fix #1

from functools import lru_cache
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter, PillowWriter
import numpy as np


from functools import lru_cache
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter, PillowWriter
import numpy as np


def _resolve_ffmpeg() -> str:
    """Resolve ffmpeg: PATH first, else IMAGEIO_FFMPEG_EXE, else imageio bundle."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        matplotlib.rcParams["animation.ffmpeg_path"] = exe
        return exe
    except Exception:
        return "ffmpeg"


def _runtime_probe(ffmpeg_exe: str, encoder_args: list[str]) -> bool:
    """Encode one 64x64 black frame to null; True iff the encoder opens successfully."""
    try:
        proc = subprocess.run(
            [
                ffmpeg_exe, "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1",
                "-frames:v", "1",
                *encoder_args,
                "-f", "null", "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        return proc.returncode == 0
    except Exception:
        return False


@lru_cache(maxsize=4)
def _is_qsv_available(ffmpeg_exe: str) -> bool:
    """Intel Quick Sync — works on most Intel iGPUs, very fast."""
    return _runtime_probe(
        ffmpeg_exe,
        ["-c:v", "h264_qsv", "-preset", "veryfast"],
    )


@lru_cache(maxsize=4)
def _is_nvenc_available(ffmpeg_exe: str) -> bool:
    """NVIDIA NVENC — physically absent on MX-series GPUs; probe just in case."""
    try:
        out = subprocess.check_output(
            [ffmpeg_exe, "-hide_banner", "-encoders"],
            stderr=subprocess.STDOUT, text=True, timeout=5,
        )
        if "h264_nvenc" not in out:
            return False
    except Exception:
        return False
    return _runtime_probe(
        ffmpeg_exe,
        ["-c:v", "h264_nvenc", "-preset", "p1"],
    )


@lru_cache(maxsize=4)
def _is_opencl_x264_available(ffmpeg_exe: str) -> bool:
    """x264 with OpenCL offload — uses MX350 CUDA cores for lookahead/motion."""
    return _runtime_probe(
        ffmpeg_exe,
        ["-c:v", "libx264", "-x264-params", "opencl=true"],
    )


def _build_mp4_writer(fps: float, use_gpu: bool = True) -> FFMpegWriter:
    """
    Build an FFMpegWriter using the best available encoder.
    Priority: QSV (Intel iGPU) → NVENC (NVIDIA) → x264+OpenCL (MX350) → x264 (CPU).
    """
    ffmpeg_exe = _resolve_ffmpeg()

    if use_gpu:
        if _is_qsv_available(ffmpeg_exe):
            print("[animator] Encoder: h264_qsv (Intel Quick Sync)")
            return FFMpegWriter(
                fps=fps,
                codec="h264_qsv",
                bitrate=2500,
                extra_args=[
                    "-preset", "veryfast",
                    "-global_quality", "23",
                    "-pix_fmt", "nv12",
                ],
            )

        if _is_nvenc_available(ffmpeg_exe):
            print("[animator] Encoder: h264_nvenc (NVIDIA)")
            return FFMpegWriter(
                fps=fps,
                codec="h264_nvenc",
                bitrate=1800,
                extra_args=[
                    "-preset", "p1",
                    "-rc", "vbr", "-cq", "28",
                    "-pix_fmt", "yuv420p",
                ],
            )

        if _is_opencl_x264_available(ffmpeg_exe):
            print("[animator] Encoder: libx264 + OpenCL (MX350 offload)")
            return FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=1800,
                extra_args=[
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-x264-params", "opencl=true",
                ],
            )

    print("[animator] Encoder: libx264 (CPU only)")
    return FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=1800,
        extra_args=[
            "-preset", "ultrafast",
            "-crf", "26",
            "-pix_fmt", "yuv420p",
        ],
    )


def _configure_ffmpeg() -> bool:
    if shutil.which("ffmpeg") is not None:
        return True
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        return True
    except Exception:
        return False


def _select_frame_indices(total_steps: int, max_frames: int) -> np.ndarray:
    if total_steps <= max_frames:
        return np.arange(total_steps, dtype=int)
    idx = np.linspace(0, total_steps - 1, max_frames)
    return np.unique(np.round(idx).astype(int))


def animate_function_fit(
    spec: ProblemSpec | None = None,
    result: dict | None = None,
    save_path_gif: Path | str | None = None,
    save_path_mp4: Path | str | None = None,
    fps: int = 10,
    max_frames: int = 300,
    target_duration_sec: float = 30.0,
    *,
    problem_code: str | None = None,
    title: str | None = None,
    x_domain: np.ndarray | None = None,
    eval_fn: Callable | None = None,
    beta_true: np.ndarray | None = None,
    beta_history: np.ndarray | None = None,
    loss_history: np.ndarray | None = None,
    event_history: list[str] | None = None,
    switch_iter: int | None = None,
    stage3_iter: int | None = None,
    save_gif_path: Path | str | None = None,
    save_mp4_path: Path | str | None = None,
    gif_path: Path | str | None = None,
    mp4_path: Path | str | None = None,
    dpi: int = 100,                                     # fix #3
    use_gpu_nvenc: bool = True,                         # fix #4
    **kwargs,
) -> None:
    _code = problem_code or (spec.code if spec is not None else "P")
    _title = title or (spec.title if spec is not None else "")
    _x = x_domain if x_domain is not None else (spec.x_domain if spec is not None else None)
    _eval_fn = eval_fn if eval_fn is not None else (spec.eval_fn if spec is not None else None)
    _beta_true = beta_true if beta_true is not None else (spec.beta_true if spec is not None else None)

    if _x is None or _eval_fn is None or _beta_true is None:
        return

    _beta_hist = (np.asarray(beta_history) if beta_history is not None
                  else np.asarray(result["beta_history"]))
    _loss_hist = (np.asarray(loss_history) if loss_history is not None
                  else np.asarray(result["loss_history"]))
    _event_hist = (event_history if event_history is not None
                   else (result.get("event_history", []) if result is not None else []))
    _switch_iter = (switch_iter if switch_iter is not None
                    else (result.get("switch_iter") if result is not None else None))
    _stage3_iter = (stage3_iter if stage3_iter is not None
                    else (result.get("stage3_iter") if result is not None else None))

    _out_gif = save_path_gif or save_gif_path or gif_path
    _out_mp4 = save_path_mp4 or save_mp4_path or mp4_path

    total_steps = len(_beta_hist)
    frame_indices = _select_frame_indices(total_steps, max_frames)
    n_frames = len(frame_indices)

    effective_fps = (max(1.0, n_frames / float(target_duration_sec))
                     if target_duration_sec and target_duration_sec > 0
                     else float(fps))
    interval_ms = 1000.0 / effective_fps

    y_true = _eval_fn(_beta_true, _x)
    y_min, y_max = float(np.min(y_true)), float(np.max(y_true))
    y_pad = 0.15 * max(y_max - y_min, 1e-3)

    iters = np.arange(total_steps)
    safe_losses = np.maximum(_loss_hist, 1e-36)
    if not _out_gif and not _out_mp4:
        return

    fig, (ax_fit, ax_loss) = plt.subplots(1, 2, figsize=(12.8, 6.4), dpi=dpi)

    ax_fit.plot(_x, y_true, color="#111827", lw=2.3,
                label="Target Sejati $f(x; \\beta^*)$", zorder=3)
    (line_est,) = ax_fit.plot(
        _x, _eval_fn(_beta_hist[0], _x), color="#2563EB", lw=2.0, ls="--",
        label="Estimasi Optimizer $f(x; \\hat{\\beta}_t)$", zorder=4,
    )
    ax_fit.set_xlim(float(_x[0]), float(_x[-1]))
    ax_fit.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_fit.set_xlabel("Domain Input ($x$)", fontsize=10.5)
    ax_fit.set_ylabel("Keluaran Model $f(x)$", fontsize=10.5)
    ax_fit.set_title("Evolusi Kecocokan Kurva", fontsize=11.5, fontweight="bold")
    ax_fit.grid(True, alpha=0.25, ls=":")
    ax_fit.legend(loc="upper right", fontsize=9.5)

    status_box = ax_fit.text(
        0.02, 0.03, "", transform=ax_fit.transAxes,
        fontsize=8.5, family="monospace", verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC",
                  edgecolor="#CBD5E1", alpha=0.92),
    )

    ax_loss.plot(iters, safe_losses, color="#CBD5E1", lw=1.2, zorder=1)
    (line_loss,) = ax_loss.plot([], [], color="#DC2626", lw=2.1, zorder=3)
    (pt_loss,) = ax_loss.plot([], [], marker="o", markersize=6,
                              color="#991B1B", zorder=4)

    if _switch_iter is not None:
        ax_loss.axvline(_switch_iter, color="#7C3AED", ls="--", lw=1.3,
                        label=f"Switch Orde-2 (Iter-{_switch_iter})")
    if _stage3_iter is not None:
        ax_loss.axvline(_stage3_iter, color="#059669", ls="-.", lw=1.3,
                        label=f"Stage-3 Freeze-Polish (Iter-{_stage3_iter})")

    ax_loss.set_yscale("log")
    ax_loss.set_xlim(0, max(total_steps - 1, 1))
    ax_loss.set_ylim(float(np.min(safe_losses)) * 0.2,
                     float(np.max(safe_losses)) * 5.0)
    ax_loss.set_xlabel("Iterasi", fontsize=10.5)
    ax_loss.set_ylabel("Loss Objektif (Log Scale)", fontsize=10.5)
    ax_loss.set_title("Trajektori Konvergensi Loss", fontsize=11.5, fontweight="bold")
    ax_loss.grid(True, which="both", alpha=0.22, ls=":")
    if _switch_iter is not None or _stage3_iter is not None:
        ax_loss.legend(loc="upper right", fontsize=9.0)

    fig.suptitle(
        f"[{_code}] {_title} (Durasi: {target_duration_sec:.0f}s | "
        f"{effective_fps:.2f} FPS)",
        fontsize=12.5, fontweight="bold", y=0.97,
    )

    def _update(frame_pos: int):
        idx = int(frame_indices[frame_pos])
        line_est.set_ydata(_eval_fn(_beta_hist[idx], _x))
        line_loss.set_data(iters[: idx + 1], safe_losses[: idx + 1])
        pt_loss.set_data([iters[idx]], [safe_losses[idx]])
        ev_text = _event_hist[idx] if idx < len(_event_hist) else ""
        status_box.set_text(
            f"Iterasi : {idx}/{total_steps - 1}\n"
            f"Loss    : {_loss_hist[idx]:.4e}\n"
            f"Fase    : {ev_text}"
        )
        return line_est, line_loss, pt_loss, status_box

    anim = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=interval_ms, blit=False,
    )

    effective_fps = max(1.0, len(frame_indices) / max(target_duration_sec, 1.0))

    if _out_mp4 is not None:                            # fix #5
        mp4_writer = _build_mp4_writer(fps=effective_fps, use_gpu=use_gpu_nvenc)
        anim.save(str(_out_mp4), writer=mp4_writer, dpi=dpi)
        print(f"Animasi MP4 disimpan di: {_out_mp4}")

    if _out_gif is not None:                            # fix #5
        gif_writer = PillowWriter(fps=effective_fps)
        anim.save(str(_out_gif), writer=gif_writer, dpi=dpi)
        print(f"Animasi GIF disimpan di: {_out_gif}")

    plt.close(fig)