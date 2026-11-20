"""Risk Weighted Assets (RWA) under Basel III/IV, the Extended Vasicek
distribution, and the standardised-approach output floor (Section 5,
Eq. 23-24).
"""

from __future__ import annotations

import numpy as np

from src.percentiles import cm_ev, cm_one_obligor, cm_vasicek_limit
from src.vasicek import bivariate_default_covariance, sigma_x_inf2, x_alpha_inf


def rwa_basel_irb(p0: float, rho: float, alpha: float, lgd: float, ead: float) -> float:
    """Basel III/IV IRB RWA (Vasicek limit): RWA = 12.5 (x_{alpha,inf}-P0) LGD EAD."""
    return 12.5 * (x_alpha_inf(p0, rho, alpha) - p0) * lgd * ead


def rwa_extended_vasicek(n: int, p0: float, rho: float, alpha: float, lgd: float, ead: float) -> float:
    """Extended Vasicek RWA (Eq. 23), serving as the RWA of a finite credit
    portfolio for alpha = 99.9%:

    RWA_{alpha,n} = 12.5 * (x_{alpha,n} - P0) * LGD * EAD

    with x_{alpha,n} the Extended Vasicek percentile of Eq. (21). This
    converges to the current Basel RWA factor in the Vasicek limit
    n -> infinity: RWA_{alpha,n} -> 12.5 * (x_{alpha,inf} - P0) * LGD * EAD.
    """
    from src.percentiles import x_alpha_extended_vasicek

    x_alpha_n = x_alpha_extended_vasicek(n, p0, rho, alpha)
    return 12.5 * (x_alpha_n - p0) * lgd * ead


def standardised_floor_rwa(risk_weight: float, ead: float, floor: float) -> float:
    """Standardised-approach RWA times the output floor: floor * risk_weight * EAD."""
    return floor * risk_weight * ead


def rwa_basel_irb_multirating(
    composition: dict[str, int],
    pd_by_rating: dict[str, float],
    rho_by_rating: dict[str, float],
    alpha: float,
    lgd: float,
    ead: float,
) -> float:
    """Basel III/IV RWA summed over rating grades (Section 5):

    RWA_{alpha,BIII/IV} = sum_I 12.5 n_I (x_{alpha,inf,I}-PD_I) LGD EAD

    This aggregation assumes perfect correlation between rating categories.
    """
    total = 0.0
    for rating, n_i in composition.items():
        total += n_i * rwa_basel_irb(pd_by_rating[rating], rho_by_rating[rating], alpha, lgd, ead)
    return total


def rwa_extended_vasicek_multirating(
    composition: dict[str, int],
    pd_by_rating: dict[str, float],
    rho_within_rating: dict[str, float],
    asset_correlation: dict[tuple[str, str], float],
    alpha: float,
    lgd: float,
    ead: float,
) -> float:
    """Analytical Extended Vasicek RWA aggregated across rating grades
    (Section 5):

    1. Per rating grade I, RWA_{alpha,n_I,I} from Eq. (23) with n = n_I.
    2. Convert the asset correlations rho_{I,J} (Table 5) into default
       correlations rho_{I,J}^def via Eq. (24):
           sigma_{I,J}^2 = Phi_2[tau_I,tau_J;rho_IJ] - P_I P_J
                         = rho_{I,J}^def sqrt(P_I(1-P_I) P_J(1-P_J))
                         = rho_{I,J}^def sigma_I sigma_J
    3. After multiplying each sigma_I with the capital multiple, EAD and LGD
       (Section 5):
           RWA_{alpha,EV} = n_Total * sqrt( sum_{I,J} rho_{I,J}^def R_I R_J )
       with R_I = f_I * RWA_{alpha,n_I,I}, f_I = n_I / n_Total.
    """
    ratings = [r for r in composition if composition[r] > 0]
    n_total = sum(composition.values())

    rwa_per_rating = {
        r: rwa_extended_vasicek(composition[r], pd_by_rating[r], rho_within_rating[r], alpha, lgd, ead)
        for r in ratings
    }
    f = {r: composition[r] / n_total for r in ratings}
    r_contribution = {r: f[r] * rwa_per_rating[r] for r in ratings}

    total = 0.0
    for i in ratings:
        for j in ratings:
            if i == j:
                rho_def_ij = 1.0
            else:
                p_i, p_j = pd_by_rating[i], pd_by_rating[j]
                rho_ij = asset_correlation[(i, j)]
                cov_ij = bivariate_default_covariance(p_i, p_j, rho_ij)
                sigma_i = np.sqrt(p_i * (1.0 - p_i))
                sigma_j = np.sqrt(p_j * (1.0 - p_j))
                rho_def_ij = cov_ij / (sigma_i * sigma_j)
            total += rho_def_ij * r_contribution[i] * r_contribution[j]
    return n_total * np.sqrt(max(total, 0.0))


def rwa_simulated(x_samples: np.ndarray, alpha: float, n_total: int, lgd: float, ead: float) -> float:
    """Simulation-based RWA (Section 5):

    RWA_{alpha,sim} = 12.5 * n_Total * (x_{alpha,sim} - <x>_sim) * LGD * EAD
    """
    from src.percentiles import harrell_davis_quantile

    x_alpha = harrell_davis_quantile(x_samples, alpha)
    mean_x = float(np.mean(x_samples))
    return 12.5 * n_total * (x_alpha - mean_x) * lgd * ead


def herfindahl_hirschman_index(composition: dict[str, int]) -> float:
    """HHI = sum_I f_I^2 with f_I the fraction of obligors in rating I (Section 5)."""
    n_total = sum(composition.values())
    return sum((n_i / n_total) ** 2 for n_i in composition.values())
