from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathConfig:
    """Mengatur seluruh direktori input/output proyek."""
    base_dir: Path = Path(__file__).resolve().parent
    output_dir: Path = base_dir / "outputs"
    plots_dir: Path = output_dir / "plots"
    reports_dir: Path = output_dir / "reports"
    animations_dir: Path = output_dir / "animations"

    def ensure_dirs(self) -> None:
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.animations_dir.mkdir(parents=True, exist_ok=True)

    def get_plot_paths(self, problem_code: str) -> tuple[Path, Path]:
        self.ensure_dirs()
        code = problem_code.lower()
        return (
            self.plots_dir / f"{code}_grid.png",
            self.plots_dir / f"{code}_combined.png",
        )

    def get_report_path(self, problem_code: str) -> Path:
        self.ensure_dirs()
        return self.reports_dir / f"{problem_code.lower()}_report.txt"

    def get_animation_paths(self, problem_code: str) -> tuple[Path, Path]:
        self.ensure_dirs()
        code = problem_code.lower()
        return (
            self.animations_dir / f"{code}_fit.gif",
            self.animations_dir / f"{code}_fit.mp4",
        )


@dataclass
class AnimationConfig:
    """Pengaturan ekspor animasi video (MP4 GPU-NVENC) dan GIF."""
    save_gif: bool = False          # Nonaktifkan GIF saat benchmark rutin (GIF murni CPU & lambat)
    save_mp4: bool = True
    use_gpu_nvenc: bool = True      # Gunakan NVIDIA NVENC 
    dpi: int = 90                   # Turunkan dari default 100-150 agar rasterisasi kanvas 2x lebih cepat
    fps: int = 10                   # 10 fps x 30 dtk = 300 frame (cukup halus & jauh lebih ringan)
    max_frames: int = 150           # Batasi maksimal 150 keyframe (efektif 5 fps untuk 30 dtk)
    target_duration_sec: float = 30.0


@dataclass
class Stage1SwarmConfig:
    """Hyperparameter Fase 1: Swarm-Adam, Cosine Decay, & Stokastik."""
    n_particles: int = 65
    min_particles_per_dim: int = 10
    max_iter: int = 600
    seed: int = 42

    lr_max: float = 0.25
    lr_min: float = 1e-5
    w_max: float = 0.78
    w_min: float = 0.28
    c1: float = 1.25
    c2: float = 1.55

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    eps: float = 1e-8
    grad_clip_norm: float = 50.0
    boundary_damping: float = 0.35

    std_tol: float = 0.025
    grad_delta_tol: float = 5e-4
    grad_norm_tol: float = 0.04
    reheat_prob: float = 0.85
    force_switch_decay: float = 0.72


@dataclass
class SensitivityConfig:
    """Hyperparameter Adaptive Hessian-Curvature & Flatness Weighting."""
    weight_power: float = 0.45
    max_param_weight: float = 100.0
    weak_weight_threshold: float = 1.8
    flat_loss_rel_tol: float = 1e-4
    subspace_relax_steps: int = 30
    subspace_relax_lr_scale: float = 0.08


@dataclass
class Stage2QuasiNewtonConfig:
    """Hyperparameter Fase 2: Multi-Basin Multi-Secant BFGS Tournament."""
    multi_basin_k: int = 6
    multi_basin_warmup_steps: int = 25


@dataclass
class Stage3SubspaceConfig:
    """Hyperparameter Fase 3: Frozen-Dominant Subspace Super-Finetuning."""
    enable_stage3: bool = True
    stage3_loss_trigger: float = 1e-24
    de_pop_per_dim: int = 18
    de_iters: int = 120
    sub_bfgs_iters: int = 80
    tr_iters: int = 60
    nm_iters: int = 150
    joint_bfgs_iters: int = 80
    joint_polish_iters: int = 60


@dataclass
class EarlyStoppingConfig:
    """Hyperparameter Weight-Gated Historical Memory Early Stopping."""
    memory_window: int = 22
    early_stop_tol: float = 1e-8
    machine_loss_tol: float = 1e-30
    stop_min_decay: float = 0.85
    min_second_order_steps: int = 18


@dataclass
class OptimizerConfig:
    """Konfigurasi utama modular yang menggabungkan seluruh komponen optimizer."""
    stage1: Stage1SwarmConfig = field(default_factory=Stage1SwarmConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    stage2: Stage2QuasiNewtonConfig = field(default_factory=Stage2QuasiNewtonConfig)
    stage3: Stage3SubspaceConfig = field(default_factory=Stage3SubspaceConfig)
    stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


# Alias kompatibilitas mundur agar skrip lama tidak rusak
ExtremeOptimizerConfig = OptimizerConfig