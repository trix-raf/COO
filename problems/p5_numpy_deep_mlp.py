import numpy as np
from problems import register_problem
from problems.base import ProblemSpec


def build_problem_p5() -> ProblemSpec:
    # 14 Parameter Jaringan Saraf Tiruan (1 -> 2 -> 2 -> 1 + Residual Skip):
    # [w1, b1, w2, b2, v11, v12, c1, v21, v22, c2, u1, u2, u_skip, d]
    beta_true = np.array(
        [
            1.65,   # beta_1 : w1 (Hidden 1 - tanh weight)
            -0.40,  # beta_2 : b1 (Hidden 1 - tanh bias)
            6.20,   # beta_3 : w2 (Hidden 1 - sin high-freq weight)
            0.85,   # beta_4 : b2 (Hidden 1 - sin phase bias)
            1.80,   # beta_5 : v11 (Hidden 2 unit 1 <- h1)
            -1.25,  # beta_6 : v12 (Hidden 2 unit 1 <- h2)
            0.30,   # beta_7 : c1  (Hidden 2 unit 1 bias)
            -1.40,  # beta_8 : v21 (Hidden 2 unit 2 <- h1, jalur lemah)
            1.95,   # beta_9 : v22 (Hidden 2 unit 2 <- h2, jalur lemah)
            -0.55,  # beta_10: c2  (Hidden 2 unit 2 bias, jalur lemah)
            2.35,   # beta_11: u1  (Output weight unit 1 - dominan)
            0.038,  # beta_12: u2  (Output weight unit 2 - gain mikro)
            -0.42,  # beta_13: u_skip (Direct residual skip connection)
            0.65,   # beta_14: d   (Output bias)
        ],
        dtype=np.float64,
    )

    x = np.linspace(-1.5, 1.5, 600)

    def _forward_cache(b: np.ndarray, x_in: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        w1, b1, w2, b2 = b[0], b[1], b[2], b[3]
        v11, v12, c1 = b[4], b[5], b[6]
        v21, v22, c2 = b[7], b[8], b[9]
        u1, u2, u_skip, d = b[10], b[11], b[12], b[13]

        # Hidden Layer 1 (Heterogeneous Basis: tanh & sin)
        z11 = w1 * x_in + b1
        h1 = np.tanh(z11)

        z12 = w2 * x_in + b2
        h2 = np.sin(z12)

        # Hidden Layer 2 (Cross-Coupled: tanh & swish)
        z21 = v11 * h1 + v12 * h2 + c1
        a1 = np.tanh(z21)

        z22 = v21 * h1 + v22 * h2 + c2
        sig22 = 1.0 / (1.0 + np.exp(-np.clip(z22, -30.0, 30.0)))
        a2 = z22 * sig22

        # Output Layer + Residual Skip
        y_hat = u1 * a1 + u2 * a2 + u_skip * x_in + d

        cache = {
            "x": x_in,
            "z12": z12,
            "h1": h1,
            "h2": h2,
            "a1": a1,
            "z22": z22,
            "sig22": sig22,
            "a2": a2,
        }
        return y_hat, cache

    def eval_fn(b: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
        y_hat, _ = _forward_cache(b, x_eval)
        return y_hat

    y_target = eval_fn(beta_true, x)

    def loss_fn(b: np.ndarray) -> float:
        y_hat, _ = _forward_cache(b, x)
        return float(0.5 * np.mean((y_hat - y_target) ** 2))

    def grad_fn(b: np.ndarray) -> np.ndarray:
        """Backpropagation analitik eksak lintas seluruh lapisan dalam NumPy murni."""
        v11, v12 = b[4], b[5]
        v21, v22 = b[7], b[8]
        u1, u2 = b[10], b[11]

        y_hat, c = _forward_cache(b, x)
        n_samples = float(len(x))
        dy = (y_hat - y_target) / n_samples

        # Gradien Output Layer
        du1 = np.sum(dy * c["a1"])
        du2 = np.sum(dy * c["a2"])
        du_skip = np.sum(dy * c["x"])
        dd = np.sum(dy)

        # Backprop ke Hidden Layer 2
        dz21 = dy * u1 * (1.0 - c["a1"] ** 2)
        swish_grad = c["sig22"] * (1.0 + c["z22"] * (1.0 - c["sig22"]))
        dz22 = dy * u2 * swish_grad

        dv11 = np.sum(dz21 * c["h1"])
        dv12 = np.sum(dz21 * c["h2"])
        dc1 = np.sum(dz21)

        dv21 = np.sum(dz22 * c["h1"])
        dv22 = np.sum(dz22 * c["h2"])
        dc2 = np.sum(dz22)

        # Backprop ke Hidden Layer 1
        dh1 = dz21 * v11 + dz22 * v21
        dh2 = dz21 * v12 + dz22 * v22

        dz11 = dh1 * (1.0 - c["h1"] ** 2)
        dz12 = dh2 * np.cos(c["z12"])

        dw1 = np.sum(dz11 * c["x"])
        db1 = np.sum(dz11)
        dw2 = np.sum(dz12 * c["x"])
        db2 = np.sum(dz12)

        return np.array(
            [dw1, db1, dw2, db2, dv11, dv12, dc1, dv21, dv22, dc2, du1, du2, du_skip, dd],
            dtype=np.float64,
        )

    return ProblemSpec(
        code="P5",
        title="Pure-NumPy 2-Hidden-Layer Heterogeneous Deep MLP + Skip (14D)",
        beta_true=beta_true,
        bounds=[
            (0.2, 3.5),               # w1
            (-2.0, 2.0),              # b1
            (2.0, 9.5),               # w2 (frekuensi tinggi lapisan 1)
            (-np.pi / 2, np.pi / 2),  # b2 (bebas aliasing fasa +/- pi)
            (-3.0, 3.0),              # v11
            (-3.0, 3.0),              # v12
            (-2.0, 2.0),              # c1
            (-2.5, 2.5),              # v21 (upstream jalur lemah)
            (-2.5, 2.5),              # v22 (upstream jalur lemah)
            (-2.0, 2.0),              # c2  (upstream jalur lemah)
            (0.5, 4.0),               # u1
            (0.005, 0.4),             # u2 (gain mikro)
            (-1.5, 1.5),              # u_skip
            (-2.0, 2.0),              # d
        ],
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        labels=[
            r"$\beta_1$ ($w_1$)",
            r"$\beta_2$ ($b_1$)",
            r"$\beta_3$ ($w_2$)",
            r"$\beta_4$ ($b_2$)",
            r"$\beta_5$ ($v_{11}$)",
            r"$\beta_6$ ($v_{12}$)",
            r"$\beta_7$ ($c_1$)",
            r"$\beta_8$ ($v_{21}$)",
            r"$\beta_9$ ($v_{22}$)",
            r"$\beta_{10}$ ($c_2$)",
            r"$\beta_{11}$ ($u_1$)",
            r"$\beta_{12}$ ($u_2$)",
            r"$\beta_{13}$ ($u_{\mathrm{skip}}$)",
            r"$\beta_{14}$ ($d$)",
        ],
        ncols=4,
        x_domain=x,
        eval_fn=eval_fn,
    )


# Daftarkan otomatis ke registry global saat modul dimuat
register_problem(build_problem_p5())