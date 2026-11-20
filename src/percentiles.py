"""Percentiles and capital multiples of the finite-portfolio loss fraction
(Section 3.2, Eq. 13, 19-21).

The Harrell-Davis quantile estimator (F.E. Harrell, C. Davis, 1982) used for
the simulated percentile x_alpha^sim is available in
scipy.stats.mstats.hdquantiles, so it is used directly rather than
reimplemented; the Extended-Vasicek percentile formula itself (Eq. 21) has no
library equivalent and is implemented from the paper.
"""

from __future__ import annotations

import numpy as np
from scipy.stats.mstats import hdquantiles

from src.moments import variance_finite
from src.vasicek import sigma_x_inf2, x_alpha_inf


def harrell_davis_quantile(samples: np.ndarray, alpha: float) -> float:
    """99.9% percentile of simulated samples via the Harrell-Davis estimator
    (Section 4.1: "the 99.9% percentile ... based on the Harrell-Davis
    estimator")."""
    return float(hdquantiles(samples, prob=[alpha])[0])


def cm_one_obligor(p0: float) -> float:
    """Capital multiple of a one-obligor portfolio (Eq. 20):
    cm_1 = sqrt((1-P0)/P0)."""
    return np.sqrt((1.0 - p0) / p0)


def cm_vasicek_limit(p0: float, rho: float, alpha: float) -> float:
    """Capital multiple in the Vasicek limit: cm_inf = (x_alpha,inf - P0)/sigma_x,inf."""
    sigma_inf = np.sqrt(sigma_x_inf2(p0, rho))
    return (x_alpha_inf(p0, rho, alpha) - p0) / sigma_inf


def cm_ev(n: int, p0: float, rho: float, alpha: float) -> float:
    """1/n-weighted capital multiple (Eq. 19):
    cm_{n,EV} = (1/n) cm_1 + (1 - 1/n) cm_inf."""
    return (1.0 / n) * cm_one_obligor(p0) + (1.0 - 1.0 / n) * cm_vasicek_limit(p0, rho, alpha)


def cm_sim(samples: np.ndarray, p0: float, alpha: float, sigma_x_n: float) -> float:
    """Simulation-based capital multiple, consistent with the defining
    relation x_alpha,n = P0 + cm_n * sigma_x,n (Eq. 13):
    cm_{n,sim} = (x_alpha^sim - P0) / sigma_x,n."""
    x_alpha_sim = harrell_davis_quantile(samples, alpha)
    return (x_alpha_sim - p0) / sigma_x_n


def x_alpha_extended_vasicek(n: int, p0: float, rho: float, alpha: float) -> float:
    """Extended Vasicek percentile (Eq. 21):

    x_{alpha,n} = P0 + { (1/n) sqrt((1-P0)/P0)
                          + (1-1/n) (x_{alpha,inf}-P0)/sigma_{x,inf} }
                  * sqrt(P0(1-P0)/n + (1-1/n) sigma_{x,inf}^2)

    x_{alpha,n} = 0 when alpha < (1-P0)^n (Section 3.2.2).
    """
    if alpha < (1.0 - p0) ** n:
        return 0.0
    cm_n = cm_ev(n, p0, rho, alpha)
    sigma_xn = np.sqrt(variance_finite(n, p0, rho))
    return p0 + cm_n * sigma_xn


def regress_cm_against_inverse_n(
    n_values: np.ndarray, cm_values: np.ndarray
) -> tuple[float, float]:
    """Linear regression of cm_{n,sim} against 1/n (Section 3.2.2, Table 4):

    cm_{n,sim} = cm_{inf,reg} + (1/n) * (cm_{1,reg} - cm_{inf,reg})

    Returns (cm_1_reg, cm_inf_reg).
    """
    inv_n = 1.0 / np.asarray(n_values, dtype=float)
    slope, intercept = np.polyfit(inv_n, cm_values, deg=1)
    cm_inf_reg = intercept
    cm_1_reg = slope + intercept
    return float(cm_1_reg), float(cm_inf_reg)
