import numpy as np
from problems import register_problem
from problems.base import ProblemSpec


def build_problem_p10() -> ProblemSpec:
    """
    10D MLP with purely piecewise-linear activations (ReLU + LeakyReLU).

    Architecture (per scalar input x):
        h1 = ReLU(w1*x + b1)
        h2 = LeakyReLU_0.05(w2*x + b2)
        z1 = v11*h1 + v12*h2 + c1
        z2 = v21*h1 + v22*h2 + c2
        y  = ReLU(z1) + LeakyReLU_0.05(z2)

    Second derivative f''(z) = 0 almost everywhere, so the Hessian
    approximation from Multi-Secant BFGS loses curvature information;
    the s^T y > 0 safeguard must rescue ProjectedDampedBFGS when it
    would otherwise fail. Dead neurons (z < 0) give exactly-zero
    gradient columns.
    """
    # [w1, b1, w2, b2, v11, v12, c1, v21, v22, c2]
    # Choosen so z1 and z2 both cross zero => ReLU output layer is
    # non-trivially exercised (some x active, some dead).
    beta_true = np.array(
        [1.2, -0.3, -0.8, 0.4, 1.0, -1.5, -0.5, -0.5, 1.2, 0.3],
        dtype=np.float64,
    )

    bounds = [
        (-4.0, 4.0),  # w1
        (-4.0, 4.0),  # b1
        (-4.0, 4.0),  # w2
        (-4.0, 4.0),  # b2
        (-3.0, 3.0),  # v11
        (-3.0, 3.0),  # v12
        (-3.0, 3.0),  # c1
        (-3.0, 3.0),  # v21
        (-3.0, 3.0),  # v22
        (-3.0, 3.0),  # c2
    ]

    x_data = np.linspace(-3.0, 3.0, 120, dtype=np.float64)
    LEAK = 0.05

    def forward(beta, x):
        w1, b1, w2, b2, v11, v12, c1, v21, v22, c2 = beta
        pre1 = w1 * x + b1
        h1 = np.maximum(pre1, 0.0)
        pre2 = w2 * x + b2
        h2 = np.where(pre2 > 0, pre2, LEAK * pre2)
        z1 = v11 * h1 + v12 * h2 + c1
        z2 = v21 * h1 + v22 * h2 + c2
        y = np.maximum(z1, 0.0) + np.where(z2 > 0, z2, LEAK * z2)
        return y, (pre1, h1, pre2, h2, z1, z2)

    y_true = forward(beta_true, x_data)[0]

    def loss_fn(beta: np.ndarray) -> float:
        y, _ = forward(beta, x_data)
        r = y - y_true
        return float(0.5 * np.mean(r ** 2))

    def grad_fn(beta: np.ndarray) -> np.ndarray:
        w1, b1, w2, b2, v11, v12, c1, v21, v22, c2 = beta
        x = x_data
        y, (pre1, h1, pre2, h2, z1, z2) = forward(beta, x)
        r = y - y_true
        n = len(x)

        # Subgradients (0 at kink) – consistent with PyTorch ReLU default.
        step_z1 = (z1 > 0).astype(np.float64)
        step_z2 = np.where(z2 > 0, 1.0, LEAK)
        step_pre1 = (pre1 > 0).astype(np.float64)
        step_pre2 = np.where(pre2 > 0, 1.0, LEAK)

        g_w1 = float(np.dot(r, step_z1 * v11 * step_pre1 * x) / n)
        g_b1 = float(np.dot(r, step_z1 * v11 * step_pre1) / n)
        g_w2 = float(np.dot(r, step_z2 * v22 * step_pre2 * x) / n)
        g_b2 = float(np.dot(r, step_z2 * v22 * step_pre2) / n)

        g_v11 = float(np.dot(r, step_z1 * h1) / n)
        g_v12 = float(np.dot(r, step_z1 * h2) / n)
        g_c1 = float(np.dot(r, step_z1) / n)

        g_v21 = float(np.dot(r, step_z2 * h1) / n)
        g_v22 = float(np.dot(r, step_z2 * h2) / n)
        g_c2 = float(np.dot(r, step_z2) / n)

        return np.array(
            [g_w1, g_b1, g_w2, g_b2, g_v11, g_v12, g_c1, g_v21, g_v22, g_c2],
            dtype=np.float64,
        )

    labels = [
        r"$w_1$", r"$b_1$", r"$w_2$", r"$b_2$",
        r"$v_{11}$", r"$v_{12}$", r"$c_1$",
        r"$v_{21}$", r"$v_{22}$", r"$c_2$",
    ]

    return ProblemSpec(
        code="P10",
        title="10D Zero-Curvature Activation MLP (ReLU + LeakyReLU)",
        beta_true=beta_true,
        bounds=bounds,
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        labels=labels,
        ncols=2,
        x_domain=x_data,
        eval_fn=lambda b, x: forward(b, x)[0],
    )


register_problem(build_problem_p10())