"""Cornish-Fisher capital multiple (Section 3.2.1, Eq. 13-18).

The paper cites (J.D. Lamb, 2016) for closed-form central moments of the
Cornish-Fisher-expanded variable but does not reproduce them. Eq. (16) defines
these moments as expectations of a standard normal variable; sympy is used
once, at import time, to expand that expectation symbolically into an exact
polynomial in (s, k) (a custom implementation of Eq. 16, not a financial
library computation), which is then compiled with sympy.lambdify into a fast
numeric function reused for every (s, k) evaluation.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import sympy as sp
import sympy.stats as sps


def _derive_standardized_moment_polynomials():
    """Derive E[(X - E[X])^3] and E[(X - E[X])^4] for
    X = Z + (Z^2-1)s + (Z^3-3Z)k - (2Z^3-5Z)s^2, Z ~ N(0,1) (Eq. 16)."""
    s, k, z = sp.symbols("s k z", real=True)
    x_expr = z + (z ** 2 - 1) * s + (z ** 3 - 3 * z) * k - (2 * z ** 3 - 5 * z) * s ** 2
    standard_normal = sps.Normal("Z", 0, 1)

    def expectation(expr):
        return sp.simplify(sps.E(expr.subs(z, standard_normal)))

    mean_x = expectation(x_expr)
    centered = sp.expand(x_expr - mean_x)
    mu3 = sp.expand(sp.simplify(expectation(centered ** 3)))
    mu4 = sp.expand(sp.simplify(expectation(centered ** 4)))
    mu3_fn = sp.lambdify((s, k), mu3, modules="numpy")
    mu4_fn = sp.lambdify((s, k), mu4, modules="numpy")
    return mu3_fn, mu4_fn


_MU3_STD_POLY, _MU4_STD_POLY = _derive_standardized_moment_polynomials()


def mu3_cf(s: float, k: float, sigma: float) -> float:
    """Third central moment mu_{3,CF}(s,k) = sigma^3 * E[(standardized X)^3]."""
    return sigma ** 3 * float(_MU3_STD_POLY(s, k))


def mu4_cf(s: float, k: float, sigma: float) -> float:
    """Fourth central moment mu_{4,CF}(s,k) = sigma^4 * E[(standardized X)^4]."""
    return sigma ** 4 * float(_MU4_STD_POLY(s, k))


def cm_cf(u_alpha: float, s: float, k: float) -> float:
    """Cornish-Fisher capital multiple (Eq. 15):

    cm_CF = u_alpha + (u_alpha^2-1)s + (u_alpha^3-3u_alpha)k
            - (2u_alpha^3-5u_alpha)s^2
    """
    return (
        u_alpha
        + (u_alpha ** 2 - 1.0) * s
        + (u_alpha ** 3 - 3.0 * u_alpha) * k
        - (2.0 * u_alpha ** 3 - 5.0 * u_alpha) * s ** 2
    )


def admissible_domain(s: float, k: float) -> bool:
    """Check whether (s, k) lie in the admissible domain R (Eq. 18), which
    guarantees that cm_CF increases monotonically with alpha. The paper's
    Eq. (18) prints the same expression for the lower and upper bound of k;
    following (J.D. Lamb, 2016), the admissible band is the interval between
    the minus and plus roots of the discriminant sqrt(s^4 - 6 s^2 + 1):

        (1 + 11 s^2 - sqrt(s^4-6s^2+1)) / 6 < k
                                 < (1 + 11 s^2 + sqrt(s^4-6s^2+1)) / 6
    """
    s2 = s ** 2
    if not (0.0 < s2 < 3.0 - 2.0 * np.sqrt(2.0)):
        return False
    discriminant = s ** 4 - 6.0 * s2 + 1.0
    if discriminant < 0:
        return False
    root = np.sqrt(discriminant)
    lower = (1.0 + 11.0 * s2 - root) / 6.0
    upper = (1.0 + 11.0 * s2 + root) / 6.0
    return lower < k < upper


def fit_sk(mu3_target: float, mu4_target: float, sigma: float) -> tuple[float, float, float]:
    """Find (s, k) minimising E(s,k) in Eq. (17) via BFGS, allowing (s, k) to
    lie outside the admissible domain R, following the paper's own fitting
    procedure in Section 3.2.1 / Section 4.2.

    mu_3,x,n and mu_4,x,n are of order sigma^3 and sigma^4, which for a
    typical portfolio loss fraction makes the raw objective E of Eq. (17) as
    small as 1e-10 to 1e-20. scipy's BFGS stops on the gradient norm of the
    objective it is given, so minimising E unscaled makes it declare
    convergence far from the optimum. E is therefore divided here by a single
    positive constant (a uniform rescaling that leaves the argmin of Eq. (17)
    unchanged) before it is handed to BFGS, purely to keep the gradient norm
    at order 1 for the optimiser's stopping rule.

    Returns (s, k, E) with E the (unscaled) minimised objective value of Eq. (17).
    """
    scale = max(sigma, 1e-6) ** 6

    def objective(params):
        s, k = params
        e3 = mu3_cf(s, k, sigma) - mu3_target
        e4 = mu4_cf(s, k, sigma) - mu4_target
        return (e3 ** 2 + e4 ** 2) / scale

    skew_guess = mu3_target / sigma ** 3 / 6.0 if sigma > 0 else 0.0
    kurt_guess = mu4_target / sigma ** 4 / 24.0 if sigma > 0 else 0.0
    result = minimize(
        objective,
        x0=[skew_guess, kurt_guess],
        method="BFGS",
        options={"gtol": 1e-14, "maxiter": 2000},
    )
    s, k = result.x
    return float(s), float(k), float(result.fun) * scale


def x_alpha_cf(p0: float, alpha: float, s: float, k: float, sigma_x_n: float) -> float:
    """Cornish-Fisher percentile x_alpha^CF = P0 + cm_CF * sigma_x,n (Eq. 13, 15)."""
    u_alpha = norm.ppf(alpha)
    return p0 + cm_cf(u_alpha, s, k) * sigma_x_n
