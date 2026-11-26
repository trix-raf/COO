import numpy as np
from problems import register_problem
from problems.base import ProblemSpec


def build_problem_p1() -> ProblemSpec:
    beta_true = np.array([3.2, -0.45, 0.0018, -0.00035, 11.5], dtype=np.float64)
    x = np.linspace(0.0, 6.0, 2500)

    def eval_fn(b: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
        e1 = np.exp(np.clip(b[1] * x_eval, -30.0, 30.0))
        e2 = np.exp(np.clip(b[3] * x_eval, -30.0, 30.0))
        return b[0] * e1 + b[2] * (x_eval ** 2) * e2 * np.cos(b[4] * x_eval)

    y = eval_fn(beta_true, x)

    def loss_fn(b: np.ndarray) -> float:
        pred = eval_fn(b, x)
        return float(0.5 * np.mean((pred - y) ** 2))

    def grad_fn(b: np.ndarray) -> np.ndarray:
        e1 = np.exp(np.clip(b[1] * x, -30.0, 30.0))
        e2 = np.exp(np.clip(b[3] * x, -30.0, 30.0))
        cos_t = np.cos(b[4] * x)
        sin_t = np.sin(b[4] * x)
        res = (b[0] * e1 + b[2] * (x ** 2) * e2 * cos_t) - y

        return np.array([
            np.mean(res * e1),
            np.mean(res * b[0] * x * e1),
            np.mean(res * (x ** 2) * e2 * cos_t),
            np.mean(res * b[2] * (x ** 3) * e2 * cos_t),
            np.mean(res * (-b[2] * (x ** 3) * e2 * sin_t)),
        ], dtype=np.float64)

    return ProblemSpec(
        code="P1",
        title="Weak-Coupling Double Exponential + High-Frequency Drift",
        beta_true=beta_true,
        bounds=[(-2.0, 6.0), (-2.0, 1.0), (-0.5, 0.5), (-0.5, 0.5), (0.0, 20.0)],
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        labels=[
            r"$\beta_1$ (Amp Utama)",
            r"$\beta_2$ (Decay Utama)",
            r"$\beta_3$ (Amp Mikro)",
            r"$\beta_4$ (Decay Mikro)",
            r"$\beta_5$ (Frekuensi)",
        ],
        ncols=3,
        x_domain=x,
        eval_fn=eval_fn,
    )


# Daftarkan otomatis ke registry global saat modul dimuat
register_problem(build_problem_p1())