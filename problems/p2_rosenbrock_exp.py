import numpy as np
from problems import register_problem
from problems.base import ProblemSpec


def build_problem_p2() -> ProblemSpec:
    beta_true = np.array([1.5, np.exp(-0.4 * 1.5), np.sin(7.0 * 1.5)], dtype=np.float64)
    x_vis = np.linspace(0.0, 3.0, 500)

    def eval_fn(b: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
        return (
            b[0] * np.cos(x_eval)
            + b[1] * np.exp(np.clip(-0.4 * x_eval, -20.0, 20.0))
            + b[2] * np.sin(7.0 * x_eval)
        )

    def loss_fn(b: np.ndarray) -> float:
        t1 = (b[0] - 1.5) ** 2
        t2 = 120.0 * (b[1] - np.exp(np.clip(-0.4 * b[0], -20.0, 20.0))) ** 2
        t3 = 0.002 * (b[2] - np.sin(7.0 * b[0])) ** 2
        return float(t1 + t2 + t3)

    def grad_fn(b: np.ndarray) -> np.ndarray:
        exp_b1 = np.exp(np.clip(-0.4 * b[0], -20.0, 20.0))
        r2 = b[1] - exp_b1
        r3 = b[2] - np.sin(7.0 * b[0])

        db1 = (
            2.0 * (b[0] - 1.5)
            + 240.0 * r2 * (0.4 * exp_b1)
            + 0.004 * r3 * (-7.0 * np.cos(7.0 * b[0]))
        )
        return np.array([db1, 240.0 * r2, 0.004 * r3], dtype=np.float64)

    return ProblemSpec(
        code="P2",
        title="Coupled Rosenbrock-Exponential + Vanishing Parameter (60.000x ratio)",
        beta_true=beta_true,
        bounds=[(-3.0, 4.0), (-1.0, 4.0), (-2.5, 2.5)],
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        labels=[r"$\beta_1$ (Coupling)", r"$\beta_2$ (Stiff Exp)", r"$\beta_3$ (Weak Sinus)"],
        ncols=3,
        x_domain=x_vis,
        eval_fn=eval_fn,
    )

# Daftarkan otomatis ke registry global saat modul dimuat
register_problem(build_problem_p2())