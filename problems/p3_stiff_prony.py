import numpy as np
from problems import register_problem
from problems.base import ProblemSpec


def build_problem_p3() -> ProblemSpec:
    beta_true = np.array([4.0, 2.8, 0.12, 0.015], dtype=np.float64)
    t = np.linspace(0.0, 15.0, 3000)

    def eval_fn(b: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        e_fast = np.exp(np.clip(-b[1] * t_eval, -30.0, 30.0))
        e_slow = np.exp(np.clip(-b[3] * t_eval, -30.0, 30.0))
        return b[0] * e_fast + b[2] * e_slow

    y = eval_fn(beta_true, t)

    def loss_fn(b: np.ndarray) -> float:
        pred = eval_fn(b, t)
        return float(0.5 * np.mean((pred - y) ** 2))

    def grad_fn(b: np.ndarray) -> np.ndarray:
        e_fast = np.exp(np.clip(-b[1] * t, -30.0, 30.0))
        e_slow = np.exp(np.clip(-b[3] * t, -30.0, 30.0))
        res = (b[0] * e_fast + b[2] * e_slow) - y

        return np.array([
            np.mean(res * e_fast),
            np.mean(res * (-b[0] * t * e_fast)),
            np.mean(res * e_slow),
            np.mean(res * (-b[2] * t * e_slow)),
        ], dtype=np.float64)

    return ProblemSpec(
        code="P3",
        title="Stiff Sum of Exponentials (Fast vs Slow Compartmental Tail)",
        beta_true=beta_true,
        bounds=[(0.5, 7.0), (0.5, 6.0), (0.0, 2.0), (0.0, 0.5)],
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        labels=[
            r"$\beta_1$ (Amp Cepat)",
            r"$\beta_2$ (Laju Cepat)",
            r"$\beta_3$ (Amp Lambat)",
            r"$\beta_4$ (Laju Lambat)",
        ],
        ncols=2,
        x_domain=t,
        eval_fn=eval_fn,
    )


# Daftarkan otomatis ke registry global saat modul dimuat
register_problem(build_problem_p3())