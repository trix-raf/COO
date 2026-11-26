import numpy as np
from problems import register_problem
from problems.base import ProblemSpec


def build_problem_p9() -> ProblemSpec:
    """
    5D micro-amplitude problem on a wide search space.

    Formulation:
        y(x) = beta_1 * sin(beta_2 * x)
             + beta_3 * exp(-((x - beta_4) ** 2) / beta_5)

    True values sit ~1e-4 .. 1e-5 while initial LHS particles explore
    bounds up to |beta| ~ 100, so the optimizer must shrink the search
    scale by ~6 orders of magnitude without stalling on early_stop_tol.
    """
    beta_true = np.array(
        [
            2.5e-5,    # beta_1 : sin amplitude (micro)
            14.2,      # beta_2 : sin frequency (fast)
            1.8e-4,    # beta_3 : gaussian amplitude (micro)
            -0.0032,   # beta_4 : gaussian center (near zero)
            0.00045,   # beta_5 : gaussian width^2 (very narrow)
        ],
        dtype=np.float64,
    )

    bounds = [
        (-50.0, 50.0),    # beta_1
        (0.1, 30.0),      # beta_2
        (-100.0, 100.0),  # beta_3
        (-50.0, 50.0),    # beta_4
        (1e-5, 25.0),     # beta_5
    ]

    x_data = np.linspace(-100.0, 5.0, 100, dtype=np.float64)

    def eval_fn(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
        b1, b2, b3, b4, b5 = beta
        sin_part = b1 * np.sin(b2 * x)
        gauss_part = b3 * np.exp(-((x - b4) ** 2) / b5)
        return sin_part + gauss_part

    y_true = eval_fn(beta_true, x_data)

    def loss_fn(beta: np.ndarray) -> float:
        r = eval_fn(beta, x_data) - y_true
        return float(0.5 * np.mean(r ** 2))

    def grad_fn(beta: np.ndarray) -> np.ndarray:
        b1, b2, b3, b4, b5 = beta
        x = x_data

        s = np.sin(b2 * x)
        c = np.cos(b2 * x)

        dx = x - b4
        e = np.exp(-(dx ** 2) / b5)

        y_hat = b1 * s + b3 * e
        r = y_hat - y_true
        inv_n = 1.0 / len(x)

        # d y / d b1 = sin(b2 x)
        g1 = float(np.dot(r, s) * inv_n)

        # d y / d b2 = b1 * x * cos(b2 x)
        g2 = float(np.dot(r, b1 * x * c) * inv_n)

        # d y / d b3 = exp(-(x-b4)^2 / b5)
        g3 = float(np.dot(r, e) * inv_n)

        # d y / d b4 = b3 * e * 2 (x - b4) / b5
        g4 = float(np.dot(r, b3 * e * (2.0 * dx / b5)) * inv_n)

        # d y / d b5 = b3 * e * (x - b4)^2 / b5^2
        g5 = float(np.dot(r, b3 * e * (dx ** 2) / (b5 ** 2)) * inv_n)

        return np.array([g1, g2, g3, g4, g5], dtype=np.float64)

    labels = [
        r"$\beta_1$ (Sin Amp)",
        r"$\beta_2$ (Sin Freq)",
        r"$\beta_3$ (Gauss Amp)",
        r"$\beta_4$ (Gauss Center)",
        r"$\beta_5$ (Gauss Width$^2$)",
    ]

    return ProblemSpec(
        code="P9",
        title="5D Micro-Amp: Sinusoid + Narrow Gaussian on Wide Range",
        beta_true=beta_true,
        bounds=bounds,
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        labels=labels,
        ncols=3,
        x_domain=x_data,
        eval_fn=eval_fn,
    )


# Daftarkan otomatis ke registry global saat modul dimuat
register_problem(build_problem_p9())