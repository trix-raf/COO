import numpy as np
from problems import register_problem
from problems.base import ProblemSpec


def build_problem_p4() -> ProblemSpec:
    beta_true = np.array([4.5, 18.0, 2.35, 0.008, 6.0, 1.45], dtype=np.float64)
    x = np.linspace(0.0, 5.0, 2500)

    def eval_fn(b: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
        denom = 1.0 + b[1] * ((x_eval - b[2]) ** 2)
        lorentz = b[0] / denom
        chirp = b[3] * np.sin(b[4] * x_eval + b[5] * (x_eval ** 2))
        return lorentz + chirp

    y = eval_fn(beta_true, x)

    def loss_fn(b: np.ndarray) -> float:
        pred = eval_fn(b, x)
        return float(0.5 * np.mean((pred - y) ** 2))

    def grad_fn(b: np.ndarray) -> np.ndarray:
        dx = x - b[2]
        denom = 1.0 + b[1] * (dx ** 2)
        inv_denom = 1.0 / denom
        inv_denom_sq = inv_denom ** 2

        phase = b[4] * x + b[5] * (x ** 2)
        sin_p = np.sin(phase)
        cos_p = np.cos(phase)

        pred = b[0] * inv_denom + b[3] * sin_p
        res = pred - y

        db1 = np.mean(res * inv_denom)
        db2 = np.mean(res * (-b[0] * (dx ** 2) * inv_denom_sq))
        db3 = np.mean(res * (2.0 * b[0] * b[1] * dx * inv_denom_sq))
        db4 = np.mean(res * sin_p)
        db5 = np.mean(res * (b[3] * x * cos_p))
        db6 = np.mean(res * (b[3] * (x ** 2) * cos_p))

        return np.array([db1, db2, db3, db4, db5, db6], dtype=np.float64)

    return ProblemSpec(
        code="P4",
        title="Lorentzian Needle Resonance + Quadratic Frequency Chirp (6D)",
        beta_true=beta_true,
        bounds=[
            (0.5, 8.0),    # beta_1: Amplitudo puncak Lorentzian
            (1.0, 35.0),   # beta_2: Ketajaman resonansi (>0 agar bebas singularitas)
            (0.2, 4.8),    # beta_3: Posisi pusat puncak
            (-0.2, 0.2),   # beta_4: Amplitudo mikro chirp
            (0.0, 12.0),   # beta_5: Frekuensi dasar chirp
            (0.0, 3.0),    # beta_6: Laju percepatan frekuensi kuadratik
        ],
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        labels=[
            r"$\beta_1$ (Amp Lorentz)",
            r"$\beta_2$ (Ketajaman)",
            r"$\beta_3$ (Pusat Puncak)",
            r"$\beta_4$ (Amp Chirp)",
            r"$\beta_5$ (Freq Dasar)",
            r"$\beta_6$ (Chirp Rate)",
        ],
        ncols=3,
        x_domain=x,
        eval_fn=eval_fn,
    )


# Daftarkan otomatis ke registry global saat modul dimuat
register_problem(build_problem_p4())