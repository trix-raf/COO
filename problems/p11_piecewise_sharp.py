import numpy as np
from problems import register_problem
from problems.base import ProblemSpec


def build_problem_p11() -> ProblemSpec:
    """
    7D piecewise-sharp / non-smooth regression.

        y(x) = b1 * |x - b2|
             + b3 * max(0, x - b4)^2
             + [ b5*(x - b6),                x <  b6
                 b7*sin(4*(x - b6)),         x >= b6 ]

    b2 is a moving absolute-value kink; b4 is a moving ReLU kink;
    b6 is a moving piecewise branch knot. Gradients jump
    discontinuously across x = b2, x = b4, x = b6, testing Armijo
    line search and Subspace-DE escape behaviour.
    """
    # b6 = 0.0 puts ~half of x_data on each branch of the piecewise term.
    beta_true = np.array(
        [0.8, 0.3, 1.5, -1.2, 0.5, 0.0, 0.6],
        dtype=np.float64,
    )

    bounds = [
        (-3.0, 3.0),  # b1
        (-2.0, 2.0),  # b2
        (-3.0, 3.0),  # b3
        (-2.5, 2.5),  # b4
        (-2.0, 2.0),  # b5
        (-2.0, 2.0),  # b6
        (-3.0, 3.0),  # b7
    ]

    x_data = np.linspace(-2.0, 2.0, 160, dtype=np.float64)

    def eval_fn(beta, x):
        b1, b2, b3, b4, b5, b6, b7 = beta
        abs_part = b1 * np.abs(x - b2)
        quad_part = b3 * np.maximum(0.0, x - b4) ** 2
        piece = np.where(x < b6, b5 * (x - b6), b7 * np.sin(4.0 * (x - b6)))
        return abs_part + quad_part + piece

    y_true = eval_fn(beta_true, x_data)

    def loss_fn(beta: np.ndarray) -> float:
        r = eval_fn(beta, x_data) - y_true
        return float(0.5 * np.mean(r ** 2))

    def grad_fn(beta: np.ndarray) -> np.ndarray:
        b1, b2, b3, b4, b5, b6, b7 = beta
        x = x_data
        n = len(x)
        r = eval_fn(beta, x) - y_true

        sx = np.sign(x - b2)                                    # ∂|x-b2|/∂b2 = -sgn
        relu_xb4 = np.maximum(0.0, x - b4)
        step_xb4 = (x > b4).astype(np.float64)                  # d/dx max(0, x-b4)
        mask_lt = (x < b6).astype(np.float64)
        mask_ge = 1.0 - mask_lt
        cos_term = 4.0 * np.cos(4.0 * (x - b6))
        sin_term = np.sin(4.0 * (x - b6))

        g1 = float(np.dot(r, np.abs(x - b2)) / n)
        g2 = float(np.dot(r, -b1 * sx) / n)
        g3 = float(np.dot(r, relu_xb4 ** 2) / n)
        g4 = float(np.dot(r, -2.0 * b3 * relu_xb4 * step_xb4) / n)
        g5 = float(np.dot(r, (x - b6) * mask_lt) / n)
        # d/db6 (piecewise) = -b5 (x<b6)  |  -4*b7*cos(4(x-b6)) (x>=b6)
        g6 = float(np.dot(r, -b5 * mask_lt - b7 * cos_term * mask_ge) / n)
        g7 = float(np.dot(r, sin_term * mask_ge) / n)

        return np.array([g1, g2, g3, g4, g5, g6, g7], dtype=np.float64)

    labels = [
        r"$\beta_1$ (Abs Amp)",
        r"$\beta_2$ (Abs Center)",
        r"$\beta_3$ (Quad Amp)",
        r"$\beta_4$ (Quad Center)",
        r"$\beta_5$ (Linear Slope)",
        r"$\beta_6$ (Knot $x$)",
        r"$\beta_7$ (Sin Amp)",
    ]

    return ProblemSpec(
        code="P11",
        title="7D Piecewise Sharp: |·| + ReLU² + Moving-Knot Piecewise",
        beta_true=beta_true,
        bounds=bounds,
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        labels=labels,
        ncols=3,
        x_domain=x_data,
        eval_fn=eval_fn,
    )


register_problem(build_problem_p11())