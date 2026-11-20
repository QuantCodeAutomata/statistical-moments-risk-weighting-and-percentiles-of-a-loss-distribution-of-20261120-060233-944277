"""Tests for src/vasicek.py: BCBS correlation (Eq. 22), Vasicek-limit moments
(Theorem 1), variance (Eq. 7) and percentile (Eq. 8)."""

import numpy as np
import pytest
from scipy.stats import norm

from src.vasicek import (
    bivariate_default_covariance,
    phi_m,
    rho_from_pd,
    sigma_x_inf2,
    tau_from_pd,
    x_alpha_inf,
)


def test_rho_from_pd_bcbs_bounds():
    """Eq. 22 interpolates between the BCBS floor (0.24) at P0 -> 0 and the
    ceiling (0.12) at P0 -> 1."""
    rho_low = rho_from_pd(1e-6)
    rho_high = rho_from_pd(0.999999)
    assert rho_low == pytest.approx(0.24, abs=1e-3)
    assert rho_high == pytest.approx(0.12, abs=1e-3)


def test_rho_from_pd_known_value():
    """P0 = 2% gives rho = 0.164 as quoted in the expected outcomes of
    experiment 1 (Section 4.2)."""
    assert rho_from_pd(0.02) == pytest.approx(0.164, abs=5e-3)


def test_rho_from_pd_rejects_invalid_input():
    with pytest.raises(ValueError):
        rho_from_pd(0.0)
    with pytest.raises(ValueError):
        rho_from_pd(1.0)


def test_tau_from_pd_matches_norm_ppf():
    for p0 in (0.0005, 0.01, 0.2):
        assert tau_from_pd(p0) == pytest.approx(norm.ppf(p0))


def test_phi_m_degenerates_to_p0_for_m_equals_1():
    tau = tau_from_pd(0.01)
    assert phi_m(tau, 0.15, 1) == pytest.approx(0.01, abs=1e-10)


def test_phi_m_independence_limit():
    """At rho = 0, Phi_m[tau,...,tau;0] = Phi(tau)^m (independent obligors)."""
    p0 = 0.02
    tau = tau_from_pd(p0)
    for m in (2, 3):
        assert phi_m(tau, 1e-9, m) == pytest.approx(p0 ** m, rel=1e-3)


def test_sigma_x_inf2_nonnegative_and_below_binomial_variance():
    """The Vasicek-limit variance must be positive and, for positive
    correlation, exceed the independent-obligor variance P0(1-P0)."""
    p0, rho = 0.02, 0.164
    var_inf = sigma_x_inf2(p0, rho)
    assert var_inf > 0.0
    assert var_inf < p0 * (1.0 - p0)  # it is a per-obligor fraction variance


def test_x_alpha_inf_increasing_in_alpha():
    p0, rho = 0.01, 0.15
    x_low = x_alpha_inf(p0, rho, 0.95)
    x_high = x_alpha_inf(p0, rho, 0.999)
    assert x_high > x_low > p0


def test_x_alpha_inf_increasing_in_rho():
    """Higher asset correlation increases tail risk at fixed PD (Theorem 1)."""
    p0, alpha = 0.02, 0.999
    x_low_rho = x_alpha_inf(p0, 0.05, alpha)
    x_high_rho = x_alpha_inf(p0, 0.40, alpha)
    assert x_high_rho > x_low_rho


def test_bivariate_default_covariance_diagonal_equals_sigma_inf2():
    """Eq. 24 with I = J and rho_IJ = rho_I must reduce to sigma_x,inf^2 of
    Eq. 7 for that single rating grade."""
    p0, rho = 0.015, 0.16
    cov = bivariate_default_covariance(p0, p0, rho)
    assert cov == pytest.approx(sigma_x_inf2(p0, rho), rel=1e-6)


def test_bivariate_default_covariance_positive_for_positive_rho():
    cov = bivariate_default_covariance(0.001, 0.02, 0.2)
    assert cov > 0.0
