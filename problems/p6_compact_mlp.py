import numpy as np
from problems import register_problem
from problems.base import ProblemSpec


def _swish(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -60.0, 60.0)
    sig = 1.0 / (1.0 + np.exp(-z_clip))
    return z * sig


def _dswish(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -60.0, 60.0)
    sig = 1.0 / (1.0 + np.exp(-z_clip))
    return sig * (1.0 + z * (1.0 - sig))


def build_problem_p6() -> ProblemSpec:
    """
    [P6] Pure-NumPy 1-Hidden-Layer Heterogeneous MLP (7D):
    Arsitektur diagnostik ringkas untuk menguji perpindahan kepentingan bobot
    antar-neuron (tanh dominan vs swish lemah 61.8x) tanpa efek multi-layer 14D:
        y(x) = u1 * tanh(w1*x + b1) + u2 * swish(w2*x + b2) + d
    """
    # [w1, b1, u1, w2, b2, u2, d]
    beta_true = np.array(
        [
            2.40,   # beta_1: w1 (input weight neuron tanh - dominan)
            -0.35,  # beta_2: b1 (bias neuron tanh - dominan)
            2.35,   # beta_3: u1 (output weight neuron tanh - dominan)
            -1.60,  # beta_4: w2 (input weight neuron swish - lemah)
            0.55,   # beta_5: b2 (bias neuron swish - lemah)
            0.038,  # beta_6: u2 (output weight neuron swish - lemah, rasio 61.8x)
            0.65,   # beta_7: d  (bias lapisan keluaran)
        ],
        dtype=np.float64,
    )

    bounds = [
        (0.5, 4.5),     # beta_1 (w1)
        (-2.0, 2.0),    # beta_2 (b1)
        (0.5, 4.5),     # beta_3 (u1)
        (-3.5, 0.5),    # beta_4 (w2)
        (-2.0, 2.0),    # beta_5 (b2)
        (-0.4, 0.4),    # beta_6 (u2)
        (-1.5, 2.5),    # beta_7 (d)
    ]

    x_data = np.linspace(-2.5, 2.5, 500, dtype=np.float64)

    def eval_fn(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
        w1, b1, u1, w2, b2, u2, d = beta
        z1 = w1 * x + b1
        a1 = np.tanh(z1)
        z2 = w2 * x + b2
        a2 = _swish(z2)
        return u1 * a1 + u2 * a2 + d

    y_true = eval_fn(beta_true, x_data)

    def loss_fn(beta: np.ndarray) -> float:
        r = eval_fn(beta, x_data) - y_true
        return float(0.5 * np.mean(r ** 2))

    def grad_fn(beta: np.ndarray) -> np.ndarray:
        w1, b1, u1, w2, b2, u2, d = beta
        z1 = w1 * x_data + b1
        a1 = np.tanh(z1)
        da1 = 1.0 - a1 ** 2

        z2 = w2 * x_data + b2
        a2 = _swish(z2)
        da2 = _dswish(z2)

        y_hat = u1 * a1 + u2 * a2 + d
        r = y_hat - y_true
        inv_n = 1.0 / len(x_data)

        g_u1 = float(np.dot(r, a1) * inv_n)
        g_u2 = float(np.dot(r, a2) * inv_n)
        g_d = float(np.sum(r) * inv_n)

        dz1 = r * u1 * da1
        g_w1 = float(np.dot(dz1, x_data) * inv_n)
        g_b1 = float(np.sum(dz1) * inv_n)

        dz2 = r * u2 * da2
        g_w2 = float(np.dot(dz2, x_data) * inv_n)
        g_b2 = float(np.sum(dz2) * inv_n)

        return np.array([g_w1, g_b1, g_u1, g_w2, g_b2, g_u2, g_d], dtype=np.float64)

    labels = [
        r"$\beta_1$ ($w_1$)",
        r"$\beta_2$ ($b_1$)",
        r"$\beta_3$ ($u_1$)",
        r"$\beta_4$ ($w_2$)",
        r"$\beta_5$ ($b_2$)",
        r"$\beta_6$ ($u_2$)",
        r"$\beta_7$ ($d$)",
    ]

    return ProblemSpec(
        code="P6",
        title="Pure-NumPy 1-Hidden-Layer Heterogeneous MLP (7D Diagnostic)",
        beta_true=beta_true,
        bounds=bounds,
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        labels=labels,
        ncols=4,
        x_domain=x_data,
        eval_fn=eval_fn,
    )


# Daftarkan otomatis ke registry global saat modul dimuat
register_problem(build_problem_p6())