import numpy as np
from problems import register_problem
from problems.base import ProblemSpec


def build_problem_p7() -> ProblemSpec:
    """
    [P7] 14D Multi-Family Hybrid Non-NN Function:
    Menggabungkan Polinom Kubik + Eksponensial + Logaritma (ln & log10) +
    Fungsi Transenden (sin & cos) + Fungsi Tangga Halus (Steep Step) dalam 14 parameter:
        f(x; beta) = (b1 + b2*x + b3*x^2 + b4*x^3)
                   + b5 * exp(-b6 * x)
                   + b7 * ln(x + b8)
                   + b9 * log10(x^2 + 1)
                   + b10 * sin(b11 * x)
                   + b12 * cos(b13 * sqrt(x))
                   + b14 / (1 + exp(-25 * (x - 2.6)))
    """
    beta_true = np.array(
        [
            1.25,     # beta_1: konstanta polinom
            -0.85,    # beta_2: koefisien linear x
            0.32,     # beta_3: koefisien kuadratik x^2
            -0.035,   # beta_4: koefisien kubik x^3
            3.40,     # beta_5: amplitudo eksponensial
            1.45,     # beta_6: laju peluruhan eksponensial
            1.15,     # beta_7: pengali logaritma natural ln(...)
            0.75,     # beta_8: pergeseran domain ln(x + b8)
            -0.90,    # beta_9: pengali log10(x^2 + 1)
            0.42,     # beta_10: amplitudo transenden sin
            3.80,     # beta_11: frekuensi transenden sin
            0.18,     # beta_12: amplitudo transenden cos(b13 * sqrt(x))
            5.20,     # beta_13: frekuensi transenden cos(b13 * sqrt(x))
            1.60,     # beta_14: tinggi lompatan fungsi tangga (step) di x = 2.6
        ],
        dtype=np.float64,
    )

    bounds = [
        (-1.0, 3.5),     # beta_1
        (-2.5, 1.5),     # beta_2
        (-0.5, 1.2),     # beta_3
        (-0.2, 0.2),     # beta_4
        (0.5, 6.5),      # beta_5
        (0.2, 3.5),      # beta_6
        (-1.0, 3.5),     # beta_7
        (0.15, 2.5),     # beta_8 (dijaga > 0 agar x + b8 > 0 selalu aman)
        (-3.0, 1.5),     # beta_9
        (-1.5, 1.5),     # beta_10
        (1.0, 7.0),      # beta_11
        (-1.0, 1.0),     # beta_12
        (1.5, 9.0),      # beta_13
        (-0.5, 4.0),     # beta_14
    ]

    x_data = np.linspace(0.05, 5.0, 600, dtype=np.float64)
    x2 = x_data ** 2
    x3 = x_data ** 3
    sqrt_x = np.sqrt(x_data)
    log10_term = np.log10(x2 + 1.0)
    step_basis = 1.0 / (1.0 + np.exp(-25.0 * (x_data - 2.6)))

    def eval_fn(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
        (
            b1, b2, b3, b4,
            b5, b6,
            b7, b8, b9,
            b10, b11, b12, b13,
            b14,
        ) = beta

        poly = b1 + b2 * x + b3 * (x ** 2) + b4 * (x ** 3)
        exp_part = b5 * np.exp(-b6 * x)
        log_part = b7 * np.log(x + b8) + b9 * np.log10((x ** 2) + 1.0)
        trans_part = b10 * np.sin(b11 * x) + b12 * np.cos(b13 * np.sqrt(x))
        step_part = b14 / (1.0 + np.exp(-25.0 * (x - 2.6)))
        return poly + exp_part + log_part + trans_part + step_part

    y_true = eval_fn(beta_true, x_data)

    def loss_fn(beta: np.ndarray) -> float:
        r = eval_fn(beta, x_data) - y_true
        return float(0.5 * np.mean(r ** 2))

    def grad_fn(beta: np.ndarray) -> np.ndarray:
        (
            b1, b2, b3, b4,
            b5, b6,
            b7, b8, b9,
            b10, b11, b12, b13,
            b14,
        ) = beta

        exp_val = np.exp(-b6 * x_data)
        log_arg = x_data + b8
        ln_val = np.log(log_arg)
        sin_val = np.sin(b11 * x_data)
        cos_val = np.cos(b11 * x_data)
        cos_sq = np.cos(b13 * sqrt_x)
        sin_sq = np.sin(b13 * sqrt_x)

        y_hat = (
            (b1 + b2 * x_data + b3 * x2 + b4 * x3)
            + b5 * exp_val
            + b7 * ln_val
            + b9 * log10_term
            + b10 * sin_val
            + b12 * cos_sq
            + b14 * step_basis
        )
        r = y_hat - y_true
        inv_n = 1.0 / len(x_data)

        g1 = float(np.sum(r) * inv_n)
        g2 = float(np.dot(r, x_data) * inv_n)
        g3 = float(np.dot(r, x2) * inv_n)
        g4 = float(np.dot(r, x3) * inv_n)

        g5 = float(np.dot(r, exp_val) * inv_n)
        g6 = float(np.dot(r, -b5 * x_data * exp_val) * inv_n)

        g7 = float(np.dot(r, ln_val) * inv_n)
        g8 = float(np.dot(r, b7 / log_arg) * inv_n)
        g9 = float(np.dot(r, log10_term) * inv_n)

        g10 = float(np.dot(r, sin_val) * inv_n)
        g11 = float(np.dot(r, b10 * x_data * cos_val) * inv_n)
        g12 = float(np.dot(r, cos_sq) * inv_n)
        g13 = float(np.dot(r, -b12 * sqrt_x * sin_sq) * inv_n)

        g14 = float(np.dot(r, step_basis) * inv_n)

        return np.array(
            [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14],
            dtype=np.float64,
        )

    labels = [
        r"$\beta_1$ (Poly $c_0$)",
        r"$\beta_2$ (Poly $x$)",
        r"$\beta_3$ (Poly $x^2$)",
        r"$\beta_4$ (Poly $x^3$)",
        r"$\beta_5$ (Exp Amp)",
        r"$\beta_6$ (Exp Rate)",
        r"$\beta_7$ ($\ln$ Gain)",
        r"$\beta_8$ ($\ln$ Shift)",
        r"$\beta_9$ ($\log_{10}$)",
        r"$\beta_{10}$ ($\sin$ Amp)",
        r"$\beta_{11}$ ($\sin$ Freq)",
        r"$\beta_{12}$ ($\cos$ Amp)",
        r"$\beta_{13}$ ($\cos$ Freq)",
        r"$\beta_{14}$ (Step Height)",
    ]

    return ProblemSpec(
        code="P7",
        title="14D Hybrid: Cubic Poly + Exp + ln/log10 + Transcendental + Step",
        beta_true=beta_true,
        bounds=bounds,
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        labels=labels,
        ncols=5,
        x_domain=x_data,
        eval_fn=eval_fn,
    )


# Daftarkan otomatis ke registry global saat modul dimuat
register_problem(build_problem_p7())