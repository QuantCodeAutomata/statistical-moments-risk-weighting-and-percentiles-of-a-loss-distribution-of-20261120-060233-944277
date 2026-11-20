"""Central moments of the loss fraction x = d/n for a finite homogeneous
credit portfolio (Section 3.1, Theorem 2, Appendix B).

scipy.special.stirling2 supplies the Stirling numbers of the second kind that
appear in Eq. (12)/(32); the moment decomposition itself (Eq. 11, Eq. 12) has
no library equivalent and is implemented directly from the paper.
"""

from __future__ import annotations

from math import comb
from functools import lru_cache

import numpy as np
from scipy.special import stirling2

from src.vasicek import phi_m, rho_from_pd, tau_from_pd


def falling_power(x: float, j: int) -> float:
    """Falling power x^{<j>} = x (x-1) ... (x-j+1), with x^{<0>} = 1
    (Appendix B, definition preceding Eq. 31)."""
    if j < 0:
        raise ValueError("j must be >= 0")
    result = 1.0
    for i in range(j):
        result *= (x - i)
    return result


def one_obligor_central_moment(m: int, p0: float) -> float:
    """m-th central moment of a one-obligor (Bernoulli(P0)) portfolio.

    x is 0 with probability (1-P0) and 1 with probability P0, so
    mu_m,x,1 = (1-P0)(0-P0)^m + P0(1-P0)^m = (1-P0)(-P0)^m + P0(1-P0)^m.
    This reduces to the binomial(n=1) central moments:
    mu_2 = P0(1-P0), mu_3 = P0(1-P0)(1-2P0), mu_4 = P0(1-P0)(1-3P0+3P0^2).
    """
    return (1.0 - p0) * (-p0) ** m + p0 * (1.0 - p0) ** m


def _noncentral_moment_vasicek(k: int, tau: float, rho: float) -> float:
    """E[x^k] in the Vasicek limit: Phi_k[tau,...,tau;rho] (Theorem 1), with
    the convention E[x^0] = 1."""
    if k == 0:
        return 1.0
    return phi_m(tau, rho, k)


def vasicek_central_moment(m: int, p0: float, rho: float) -> float:
    """m-th central moment (mu_m)_{x,inf} of the Vasicek distribution, obtained
    from its non-central moments (Theorem 1) via the standard binomial
    expansion mu_m = sum_{k=0}^{m} C(m,k) (-P0)^{m-k} E[x^k]."""
    tau = tau_from_pd(p0)
    return sum(
        comb(m, k) * (-p0) ** (m - k) * _noncentral_moment_vasicek(k, tau, rho)
        for k in range(m + 1)
    )


def b_kl(n: float, k: int, l: int) -> float:
    """Coefficient b_{k,l} of Eq. (32)/Table 3:
    b_{k,l} = S_{k,l} * (n-2)^{<l-2>} / n^{k-2}.
    """
    s_kl = float(stirling2(k, l))
    return s_kl * falling_power(n - 2, l - 2) / n ** (k - 2)


def a_ml(m: int, l: int, n: int, p0: float) -> float:
    """Coefficient a_{m,l} of Eq. (12) in the decomposition of Theorem 2
    (Eq. 11). Verified against Table 2 for m = 3 and m = 4 in the test suite.
    """
    if l == 1:
        total = 0.0
        for k in range(2, m + 1):
            total += comb(m, k) * (-p0) ** (m - k) * (n ** (-(k - 2)) - 1.0) / (n - 1.0)
        return n * total
    total = 0.0
    for k in range(l, m + 1):
        total += comb(m, k) * (-p0) ** (m - k) * b_kl(n, k, l)
    total -= comb(m, l) * (-p0) ** (m - l)
    return n * total


def central_moment_finite(m: int, n: int, p0: float, rho: float) -> float:
    """Central moment (mu_m)_{x,n} of a finite portfolio of n obligors
    (Theorem 2, Eq. 11):

    (mu_m)_{x,n} = (1/n) (mu_m)_{x,1} + (1 - 1/n) (mu_m)_{x,inf}
                   + (1/n)(1 - 1/n) * sum_{l=1}^{m} a_{m,l} Phi_l[tau,...,tau;rho]
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    tau = tau_from_pd(p0)
    mu_1 = one_obligor_central_moment(m, p0)
    mu_inf = vasicek_central_moment(m, p0, rho)
    if n == 1:
        return mu_1
    correction = sum(a_ml(m, l, n, p0) * phi_m(tau, rho, l) for l in range(1, m + 1))
    return (1.0 / n) * mu_1 + (1.0 - 1.0 / n) * mu_inf + (1.0 / n) * (1.0 - 1.0 / n) * correction


def variance_finite(n: int, p0: float, rho: float) -> float:
    """Variance sigma_x,n^2 of a finite portfolio (Eq. 10):

    sigma_x,n^2 = P0(1-P0)/n + (1 - 1/n) * sigma_x,inf^2
    """
    from src.vasicek import sigma_x_inf2

    if n < 1:
        raise ValueError("n must be >= 1")
    sigma_inf2 = sigma_x_inf2(p0, rho)
    return p0 * (1.0 - p0) / n + (1.0 - 1.0 / n) * sigma_inf2
