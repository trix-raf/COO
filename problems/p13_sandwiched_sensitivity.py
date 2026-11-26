import numpy as np
from problems import register_problem
from problems.base import ProblemSpec


def build_problem_p13() -> ProblemSpec:
    """
    6D nested radicals with a *sandwiched-sensitivity* trap at beta_5.

        f(x) = sqrt( b1*x + sqrt( b2*x + sqrt( b3*x + sqrt( b4*x
                    + sqrt( b5*x + sqrt( b6*x ) ) ) ) ) )

    beta_true = [1.0, 0.2, 0.05, 0.007, 9e-4, 3e-5]

    Why beta_5 is the hardest parameter (not beta_6, despite being ~30x
    larger than beta_6):
        Each inner layer's linear term beta_k*x only begins to dominate
        the next-nested sqrt when
            x > x*_k = beta_{k+1} / beta_k^2
        Cross-over points on beta_true:
            x*_1 = 0.20, x*_2 = 1.25, x*_3 = 2.80,
            x*_4 = 18.37, x*_5 = 37.04
        So beta_5's "own window" is only x in [18.37, 37.04].  On the
        wrong x-grid (e.g. linspace(0.01, 5.0)) this window is never
        touched, beta_5 becomes almost unobservable, and Cond(J) blows
        out to ~4e6.  Using geomspace(1e-4, 100.0) keeps Cond(J) ~2.6e5
        while all five cross-over points remain identifiable.

    Relative column norms ||J_{:,k} * beta_k||_2 on geomspace:
        [70.38, 2.65, 0.326, 0.0607, **0.0476**, 0.495]
    i.e. a *valley* at index 4 (beta_5), not the deepest beta_6.  This
    is the numerical trap that SensitivityWeighter must survive: if it
    naively weights by |beta| it will mis-rank beta_6 as the weakest.
    """
    beta_true = np.array(
        [1.0, 0.2, 0.05, 0.007, 0.0009, 0.00003],
        dtype=np.float64,
    )

    # Strictly positive bounds keep every sqrt argument real throughout
    # LHS sampling and hybrid re-heat.  Wide upper ranges force the
    # swarm to climb the deepest radical face.
    bounds = [
        (0.05, 8.0),      # beta_1 (true: 1.0)
        (0.005, 5.0),     # beta_2 (true: 0.2)
        (0.001, 3.0),     # beta_3 (true: 0.05)
        (0.0001, 1.5),    # beta_4 (true: 0.007)
        (1e-5, 0.5),      # beta_5 (true: 0.0009)  <- sandwiched
        (1e-6, 0.2),      # beta_6 (true: 3e-5)
    ]

    # Logarithmic domain — mandatory for beta_5 to be identifiable.
    x_data = np.geomspace(1e-20, 100.0, 3000, dtype=np.float64)

    TINY = 1e-300  # guard against division-by-zero in backward pass

    def _forward_and_jac(beta: np.ndarray, x: np.ndarray):
        """Analytic forward pass + Jacobian for the 6-layer nested sqrt."""
        b1, b2, b3, b4, b5, b6 = beta

        # Forward pass: innermost (index 5) -> outermost (index 0).
        # np.maximum(..., 0.0) is defensive; positive bounds guarantee
        # it is a no-op during valid optimization trajectories.
        u = [None] * 6
        u[5] = np.sqrt(np.maximum(b6 * x, 0.0))
        u[4] = np.sqrt(np.maximum(b5 * x + u[5], 0.0))
        u[3] = np.sqrt(np.maximum(b4 * x + u[4], 0.0))
        u[2] = np.sqrt(np.maximum(b3 * x + u[3], 0.0))
        u[1] = np.sqrt(np.maximum(b2 * x + u[2], 0.0))
        u[0] = np.sqrt(np.maximum(b1 * x + u[1], 0.0))

        # Backward pass (chain rule through sqrt layers).
        # d f / d beta_k = x / ( 2^{k+1} * prod_{j=0..k} u_j )
        J = np.empty((x.size, 6), dtype=np.float64)
        denom = np.ones_like(x)
        for k in range(6):
            denom = denom * (2.0 * np.maximum(u[k], TINY))
            J[:, k] = x / denom
        return u[0], J

    y_true = _forward_and_jac(beta_true, x_data)[0]

    def eval_fn(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return _forward_and_jac(beta, x)[0]

    def loss_fn(beta: np.ndarray) -> float:
        r = _forward_and_jac(beta, x_data)[0] - y_true
        return float(0.5 * np.mean(r ** 2))

    def grad_fn(beta: np.ndarray) -> np.ndarray:
        f, J = _forward_and_jac(beta, x_data)
        r = f - y_true
        n = x_data.size
        # dL/dbeta_k = (1/n) * sum_i r_i * J_{i,k}
        return (J * r[:, None]).sum(axis=0) / n

    labels = [
        r"$\beta_1$ (outermost)",
        r"$\beta_2$",
        r"$\beta_3$",
        r"$\beta_4$",
        r"$\beta_5$ (sandwiched)",
        r"$\beta_6$ (innermost)",
    ]

    return ProblemSpec(
        code="P13",
        title="6D Nested Radicals: Sandwiched-Sensitivity Trap",
        beta_true=beta_true,
        bounds=bounds,
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        labels=labels,
        ncols=3,
        x_domain=x_data,
        eval_fn=eval_fn,
    )


register_problem(build_problem_p13())