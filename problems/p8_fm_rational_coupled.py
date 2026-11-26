import numpy as np
from problems import register_problem
from problems.base import ProblemSpec


def build_problem_p8() -> ProblemSpec:
    """
    [P8] 12D Non-Separable FM-Gaussian Wavepacket + Rational Peak + Hyperbolic Saturation:
    Sistem nonlinier bersarang penuh tanpa pemisahan linier trivial:
        f(x; beta) = b1 * exp(-(x - b2)^2 / (2 * b3^2)) * cos(b4 * x + b5 * sin(b6 * x))
                   + (b7 * x + b8) / (1 + b9 * (x - b10)^2)
                   + b11 * tanh(b12 * x - 1.5)
    """
    beta_true = np.array(
        [
            2.80,    # beta_1: amplitudo wavepacket FM-Gaussian
            1.75,    # beta_2: pusat selubung Gaussian
            0.32,    # beta_3: lebar selubung Gaussian (sempit)
            9.50,    # beta_4: frekuensi pembawa (carrier)
            1.35,    # beta_5: indeks modulasi fase FM
            2.60,    # beta_6: frekuensi modulasi FM
            0.48,    # beta_7: kemiringan pembilang rasional
            1.65,    # beta_8: intersep pembilang rasional
            4.20,    # beta_9: ketajaman penyebut rasional
            3.40,    # beta_10: pusat resonansi rasional
            0.045,   # beta_11: amplitudo mikro saturasi tanh (~62x lebih kecil)
            3.10,    # beta_12: kecuraman transisi tanh
        ],
        dtype=np.float64,
    )

    bounds = [
        (0.5, 5.0),     # beta_1
        (0.8, 2.8),     # beta_2
        (0.12, 0.8),    # beta_3
        (5.0, 14.0),    # beta_4
        (0.2, 3.0),     # beta_5
        (1.0, 4.5),     # beta_6
        (-0.5, 1.5),    # beta_7
        (0.2, 3.5),     # beta_8
        (1.0, 8.0),     # beta_9
        (2.2, 4.5),     # beta_10
        (-0.4, 0.4),    # beta_11
        (1.0, 5.5),     # beta_12
    ]

    x_data = np.linspace(0.0, 5.0, 650, dtype=np.float64)

    def eval_fn(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
        b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12 = beta

        dx_g = x - b2
        inv_b3_sq = 1.0 / (b3 ** 2)
        gauss_env = np.exp(-0.5 * (dx_g ** 2) * inv_b3_sq)

        mod_sin = np.sin(b6 * x)
        phase = b4 * x + b5 * mod_sin
        carrier = np.cos(phase)

        dx_r = x - b10
        denom_r = 1.0 + b9 * (dx_r ** 2)
        rational_part = (b7 * x + b8) / denom_r

        tanh_part = b11 * np.tanh(b12 * x - 1.5)

        return b1 * gauss_env * carrier + rational_part + tanh_part

    y_true = eval_fn(beta_true, x_data)

    def loss_fn(beta: np.ndarray) -> float:
        r = eval_fn(beta, x_data) - y_true
        return float(0.5 * np.mean(r ** 2))

    def grad_fn(beta: np.ndarray) -> np.ndarray:
        b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12 = beta

        dx_g = x_data - b2
        inv_b3_sq = 1.0 / (b3 ** 2)
        gauss_env = np.exp(-0.5 * (dx_g ** 2) * inv_b3_sq)

        mod_sin = np.sin(b6 * x_data)
        mod_cos = np.cos(b6 * x_data)
        phase = b4 * x_data + b5 * mod_sin
        carrier = np.cos(phase)
        d_carrier = -np.sin(phase)

        wave = gauss_env * carrier
        d_phase_common = b1 * gauss_env * d_carrier

        dx_r = x_data - b10
        num_r = b7 * x_data + b8
        denom_r = 1.0 + b9 * (dx_r ** 2)
        inv_denom_r = 1.0 / denom_r
        rational_part = num_r * inv_denom_r

        th_val = np.tanh(b12 * x_data - 1.5)
        d_th = 1.0 - th_val ** 2

        y_hat = b1 * wave + rational_part + b11 * th_val
        r = y_hat - y_true
        inv_n = 1.0 / len(x_data)

        g1 = float(np.dot(r, wave) * inv_n)
        g2 = float(np.dot(r, b1 * wave * (dx_g * inv_b3_sq)) * inv_n)
        g3 = float(np.dot(r, b1 * wave * ((dx_g ** 2) / (b3 ** 3))) * inv_n)
        g4 = float(np.dot(r, d_phase_common * x_data) * inv_n)
        g5 = float(np.dot(r, d_phase_common * mod_sin) * inv_n)
        g6 = float(np.dot(r, d_phase_common * b5 * x_data * mod_cos) * inv_n)

        g7 = float(np.dot(r, x_data * inv_denom_r) * inv_n)
        g8 = float(np.dot(r, inv_denom_r) * inv_n)
        g9 = float(np.dot(r, -rational_part * (dx_r ** 2) * inv_denom_r) * inv_n)
        g10 = float(np.dot(r, rational_part * (2.0 * b9 * dx_r) * inv_denom_r) * inv_n)

        g11 = float(np.dot(r, th_val) * inv_n)
        g12 = float(np.dot(r, b11 * x_data * d_th) * inv_n)

        return np.array(
            [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12],
            dtype=np.float64,
        )

    labels = [
        r"$\beta_1$ (FM Amp)",
        r"$\beta_2$ (Env $\mu$)",
        r"$\beta_3$ (Env $\sigma$)",
        r"$\beta_4$ (Carrier $\omega_c$)",
        r"$\beta_5$ (FM Index)",
        r"$\beta_6$ (Mod $\omega_m$)",
        r"$\beta_7$ (Rat Slope)",
        r"$\beta_8$ (Rat Bias)",
        r"$\beta_9$ (Rat Sharp)",
        r"$\beta_{10}$ (Rat Center)",
        r"$\beta_{11}$ ($\tanh$ Micro)",
        r"$\beta_{12}$ ($\tanh$ Gain)",
    ]

    return ProblemSpec(
        code="P8",
        title="12D Non-Separable FM-Gaussian Wavepacket + Rational + Tanh",
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
register_problem(build_problem_p8())