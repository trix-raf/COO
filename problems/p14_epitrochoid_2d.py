import numpy as np
from problems import register_problem
from problems.base import ProblemSpec


def build_problem_p14() -> ProblemSpec:
    """
    P14: 2D Epitrochoid (Phase-Modulated Ellipse), 7D.

    Parametric curve over t in [0, 2*pi):
        x(t) = c_x + a*cos(t + phi1) + d*cos(3t + phi2)
        y(t) = c_y + b*sin(t + phi1) + d*sin(3t + phi2)

    Tests:
      * Stacked 2D vector regression: eval_fn returns a length-2N
        vector (x|y concatenated), analytic Jacobian (2N x 7).
      * Phase aliasing avoided by phi1, phi2 in (-pi, pi) and
        a, b, d > 0 (otherwise (a, phi1) ~ (-a, phi1 + pi), etc.).
      * grad_fn consistent with 0.5 * mean(r^2) over 2N residuals.

    Run with --no-gif --no-mp4 (default animator expects 1D y(x)).
    """
    beta_true = np.array(
        [1.5, -0.8, 2.0, 1.2, 0.7, 0.0, np.pi / 4.0],
        dtype=np.float64,
    )
    # order: c_x, c_y, a, b, d, phi1, phi2

    bounds = [
        (-4.0, 4.0),         # c_x
        (-4.0, 4.0),         # c_y
        (0.2, 4.0),          # a  (positive -> resolves sign/phase ambiguity)
        (0.2, 4.0),          # b
        (0.1, 2.0),          # d  (positive -> unique phi2)
        (-np.pi, np.pi),     # phi1
        (-np.pi, np.pi),     # phi2
    ]

    N = 400
    t = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    x_domain = np.arange(2 * N, dtype=np.float64)  # dummy length marker

    def _eval_jac(beta):
        cx, cy, a, b, d, p1, p2 = beta

        ang1 = t + p1
        ang3 = 3.0 * t + p2
        c1, s1 = np.cos(ang1), np.sin(ang1)
        c3, s3 = np.cos(ang3), np.sin(ang3)

        x = cx + a * c1 + d * c3
        y = cy + b * s1 + d * s3

        zeros = np.zeros(N)
        ones = np.ones(N)

        # dx/d[cx, cy, a, b, d, phi1, phi2]
        Jx = np.column_stack([
            ones, zeros,
            c1, zeros,
            c3,
            -a * s1,
            -3.0 * d * s3,
        ])
        # dy/d[cx, cy, a, b, d, phi1, phi2]
        Jy = np.column_stack([
            zeros, ones,
            zeros, s1,
            s3,
            b * c1,
            3.0 * d * c3,
        ])

        J = np.vstack([Jx, Jy])  # (2N, 7)
        return np.concatenate([x, y]), J

    y_true, _ = _eval_jac(beta_true)

    def eval_fn(beta, _x):
        return _eval_jac(beta)[0]

    def loss_fn(beta):
        r = eval_fn(beta, x_domain) - y_true
        return float(0.5 * np.mean(r ** 2))

    def grad_fn(beta):
        y, J = _eval_jac(beta)
        r = y - y_true
        return (J.T @ r) / (2.0 * N)

    labels = [
        r"$c_x$",
        r"$c_y$",
        r"$a$ (cos amp)",
        r"$b$ (sin amp)",
        r"$d$ (3rd-harm amp)",
        r"$\phi_1$",
        r"$\phi_2$",
    ]

    return ProblemSpec(
        code="P14",
        title="7D 2D Epitrochoid: Phase-Modulated Ellipse + 3rd Harmonic",
        beta_true=beta_true,
        bounds=bounds,
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        labels=labels,
        ncols=3,
        x_domain=x_domain,
        eval_fn=eval_fn,
    )


register_problem(build_problem_p14())