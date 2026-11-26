import numpy as np
from problems import register_problem
from problems.base import ProblemSpec


def build_problem_p15() -> ProblemSpec:
    """
    P15: 3D Torus Knot, 9D.

    Curve over t in [0, 2*pi):
        radial(t) = R + r*cos(q*t + psi)
        x(t) = c_x + radial * cos(p*t + phi)
        y(t) = c_y + radial * sin(p*t + phi)
        z(t) = c_z + r*sin(q*t + psi)

    With true (p, q) = (3, 2), this traces the trefoil torus knot.
    p and q are bounded in narrow windows (2.7-3.3, 1.7-2.3) because
    for *continuous* (p, q) the curve is only exactly closed at
    integer values, producing a sharp minimum that is identifiable
    but non-convex enough to stress Armijo line search.

    Tests:
      * Stacked 3D vector regression: 3N = 1500 residuals vs 9 params.
      * Analytic Jacobian with frequency parameters (p, q) coupling
        to every other term via the winding argument - large
        magnitude spread between columns (d/dc ~ 1, d/dp ~ radial*t ~ 40).
      * Rotation/phase aliasing suppressed by phi, psi in (-pi, pi).
      * R, r > 0 to avoid (R, r) -> (-R, -r) reflection.

    Run with --no-gif --no-mp4 (default animator expects 1D y(x)).
    """
    beta_true = np.array(
        [0.5, -0.3, 1.0, 2.0, 0.8, 3.0, 2.0, 0.0, np.pi / 4.0],
        dtype=np.float64,
    )
    # order: c_x, c_y, c_z, R, r, p, q, phi, psi

    bounds = [
        (-2.0, 2.0),         # c_x
        (-2.0, 2.0),         # c_y
        (0.2, 2.0),          # c_z
        (1.0, 3.5),          # R
        (0.2, 1.5),          # r
        (2.7, 3.3),          # p  (winding around z-axis, near 3)
        (1.7, 2.3),          # q  (torus poloidal winding, near 2)
        (-np.pi, np.pi),     # phi
        (-np.pi, np.pi),     # psi
    ]

    N = 500
    t = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    x_domain = np.arange(3 * N, dtype=np.float64)

    def _eval_jac(beta):
        cx, cy, cz, R, r, p, q, phi, psi = beta

        ang_p = p * t + phi
        ang_q = q * t + psi
        c_p, s_p = np.cos(ang_p), np.sin(ang_p)
        c_q, s_q = np.cos(ang_q), np.sin(ang_q)
        radial = R + r * c_q

        x = cx + radial * c_p
        y = cy + radial * s_p
        z = cz + r * s_q

        zeros = np.zeros(N)
        ones = np.ones(N)

        # --- x derivatives ---------------------------------------------
        Jx = np.column_stack([
            ones, zeros, zeros,                   # cx, cy, cz
            c_p,                                  # R
            c_q * c_p,                            # r
            radial * (-s_p) * t,                  # p
            r * (-s_q) * t * c_p,                 # q
            radial * (-s_p),                      # phi
            -r * s_q * c_p,                       # psi
        ])
        # --- y derivatives ---------------------------------------------
        Jy = np.column_stack([
            zeros, ones, zeros,
            s_p,
            c_q * s_p,
            radial * c_p * t,
            r * (-s_q) * t * s_p,
            radial * c_p,
            -r * s_q * s_p,
        ])
        # --- z derivatives ---------------------------------------------
        Jz = np.column_stack([
            zeros, zeros, ones,
            zeros,
            s_q,
            zeros,
            r * c_q * t,
            zeros,
            r * c_q,
        ])

        J = np.vstack([Jx, Jy, Jz])  # (3N, 9)
        return np.concatenate([x, y, z]), J

    y_true, _ = _eval_jac(beta_true)

    def eval_fn(beta, _x):
        return _eval_jac(beta)[0]

    def loss_fn(beta):
        r = eval_fn(beta, x_domain) - y_true
        return float(0.5 * np.mean(r ** 2))

    def grad_fn(beta):
        y, J = _eval_jac(beta)
        r = y - y_true
        return (J.T @ r) / (3.0 * N)

    labels = [
        r"$c_x$",
        r"$c_y$",
        r"$c_z$",
        r"$R$",
        r"$r$",
        r"$p$ (axial winding)",
        r"$q$ (poloidal winding)",
        r"$\phi$",
        r"$\psi$",
    ]

    return ProblemSpec(
        code="P15",
        title="9D 3D Torus Knot (p, q): Stacked 3N Space Curve",
        beta_true=beta_true,
        bounds=bounds,
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        labels=labels,
        ncols=3,
        x_domain=x_domain,
        eval_fn=eval_fn,
    )


register_problem(build_problem_p15())