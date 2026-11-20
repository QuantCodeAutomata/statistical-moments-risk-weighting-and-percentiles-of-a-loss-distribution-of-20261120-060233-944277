"""Vasicek-limit (n -> infinity) building blocks.

Implements the BCBS correlation formula (Eq. 22), the single-factor default
threshold, the non-central Vasicek moments of Theorem 1 (Eq. 5), the Vasicek
variance (Eq. 7) and the Vasicek percentile (Eq. 8). scipy.stats is used for
the univariate and multivariate normal distribution functions; no
quantitative finance library offers these primitives, so the formulas
themselves follow Theorem 1 directly.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.stats import multivariate_normal, norm


def rho_from_pd(p0: float) -> float:
    """BCBS asset-correlation formula (Eq. 22, Section 4.1).

    rho = 0.12 * (1 - exp(-50 P0)) / (1 - exp(-50))
          + 0.24 * (1 - (1 - exp(-50 P0)) / (1 - exp(-50)))
    """
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"P0 must be in (0, 1), got {p0}")
    w = (1.0 - np.exp(-50.0 * p0)) / (1.0 - np.exp(-50.0))
    return 0.12 * w + 0.24 * (1.0 - w)


def tau_from_pd(p0: float) -> float:
    """Default threshold tau = Phi^{-1}[P0] (Section 2, Eq. 3)."""
    return norm.ppf(p0)


def _homogeneous_correlation_matrix(m: int, rho: float) -> np.ndarray:
    """The m x m homogeneous correlation matrix C_m (Eq. 2): unit diagonal,
    off-diagonal entries equal to rho."""
    c = np.full((m, m), rho, dtype=float)
    np.fill_diagonal(c, 1.0)
    return c


@lru_cache(maxsize=4096)
def phi_m(tau: float, rho: float, m: int) -> float:
    """Non-central Vasicek moment Phi_m[tau,...,tau; rho] of Theorem 1 (Eq. 5):
    the m-dimensional cumulative standard normal probability with homogeneous
    correlation matrix C_m, evaluated at (tau, ..., tau).

    For m == 1 this is simply Phi(tau) = P0. For m >= 2 the cumulative
    multivariate normal distribution (scipy.stats.multivariate_normal.cdf) is
    used, since no closed form exists for m > 1.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if m == 1:
        return float(norm.cdf(tau))
    mean = np.zeros(m)
    cov = _homogeneous_correlation_matrix(m, rho)
    point = np.full(m, tau)
    return float(multivariate_normal.cdf(point, mean=mean, cov=cov))


def sigma_x_inf2(p0: float, rho: float) -> float:
    """Vasicek-limit variance sigma_x,inf^2 = Phi_2[tau,tau;rho] - P0^2 (Eq. 7)."""
    tau = tau_from_pd(p0)
    return phi_m(tau, rho, 2) - p0 ** 2


def x_alpha_inf(p0: float, rho: float, alpha: float) -> float:
    """Vasicek-limit percentile (Eq. 8):

    x_{alpha,inf} = Phi[(tau + sqrt(rho) * Phi^{-1}[alpha]) / sqrt(1 - rho)]
    """
    tau = tau_from_pd(p0)
    u_alpha = norm.ppf(alpha)
    return float(norm.cdf((tau + np.sqrt(rho) * u_alpha) / np.sqrt(1.0 - rho)))


def bivariate_default_covariance(p_i: float, p_j: float, rho_ij: float) -> float:
    """sigma_{I,J}^2 = Phi_2[Phi^{-1}(P_I), Phi^{-1}(P_J); rho_IJ] - P_I * P_J
    (Eq. 24), the joint default covariance between two (possibly different)
    rating grades I and J with asset correlation rho_IJ.
    """
    tau_i, tau_j = norm.ppf(p_i), norm.ppf(p_j)
    cov = np.array([[1.0, rho_ij], [rho_ij, 1.0]])
    joint = float(multivariate_normal.cdf([tau_i, tau_j], mean=[0.0, 0.0], cov=cov))
    return joint - p_i * p_j
