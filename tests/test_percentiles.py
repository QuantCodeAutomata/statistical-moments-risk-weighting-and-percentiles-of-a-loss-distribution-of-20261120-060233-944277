"""Tests for src/percentiles.py: capital multiples and the Extended Vasicek
percentile (Section 3.2, Eq. 13, 19-21)."""

import numpy as np
import pytest

from src.percentiles import (
    cm_ev,
    cm_one_obligor,
    cm_sim,
    cm_vasicek_limit,
    harrell_davis_quantile,
    regress_cm_against_inverse_n,
    x_alpha_extended_vasicek,
)
from src.moments import variance_finite
from src.vasicek import rho_from_pd, x_alpha_inf


def test_harrell_davis_quantile_matches_known_percentile():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(200_000)
    q = harrell_davis_quantile(samples, 0.999)
    from scipy.stats import norm

    assert q == pytest.approx(norm.ppf(0.999), abs=0.02)


def test_cm_one_obligor_formula():
    p0 = 0.04
    assert cm_one_obligor(p0) == pytest.approx(np.sqrt((1 - p0) / p0))


def test_cm_ev_interpolates_between_cm1_and_cm_inf():
    p0, rho, alpha = 0.02, 0.164, 0.999
    cm1 = cm_one_obligor(p0)
    cm_inf = cm_vasicek_limit(p0, rho, alpha)
    cm_n = cm_ev(50, p0, rho, alpha)
    lo, hi = sorted((cm1, cm_inf))
    assert lo <= cm_n <= hi
    # Eq. 19: cm_n -> cm_1 as n -> 1 and cm_n -> cm_inf as n -> infinity.
    assert cm_ev(1, p0, rho, alpha) == pytest.approx(cm1)
    assert cm_ev(1_000_000, p0, rho, alpha) == pytest.approx(cm_inf, rel=1e-4)


def test_x_alpha_extended_vasicek_converges_to_vasicek_limit():
    p0 = 0.02
    rho = rho_from_pd(p0)
    alpha = 0.999
    x_large_n = x_alpha_extended_vasicek(5000, p0, rho, alpha)
    x_vasicek = x_alpha_inf(p0, rho, alpha)
    assert x_large_n == pytest.approx(x_vasicek, rel=1e-2)


def test_x_alpha_extended_vasicek_zero_below_threshold():
    """x_{alpha,n} = 0 when alpha < (1-P0)^n (Section 3.2.2)."""
    p0, rho, alpha = 0.3, 0.2, 0.5
    n = 2  # (1-0.3)^2 = 0.49 < alpha = 0.5 is false; pick n with (1-p0)^n > alpha
    n = 5  # (0.7)^5 = 0.168 < 0.5, so this is not the zero regime; use direct check
    threshold_n = 1  # (1-p0)^1 = 0.7 > alpha=0.5 -> not zero
    # Choose a combination that explicitly satisfies alpha < (1-p0)^n.
    p0_small, alpha_small, n_small = 0.3, 0.1, 1  # (1-0.3)^1 = 0.7 > 0.1 -> zero regime
    assert alpha_small < (1 - p0_small) ** n_small
    assert x_alpha_extended_vasicek(n_small, p0_small, rho, alpha_small) == 0.0


def test_x_alpha_extended_vasicek_closer_to_simulation_than_vasicek_for_small_n():
    """Expected outcome of experiment 3: the Extended Vasicek percentile is
    considerably closer to the simulated percentile than the plain Vasicek
    percentile for small n."""
    from data.generate_synthetic import simulate_fraction_defaults

    p0 = 0.02
    rho = rho_from_pd(p0)
    alpha = 0.999
    n = 2
    rng = np.random.default_rng(1)
    x_sim = harrell_davis_quantile(
        simulate_fraction_defaults(p0, rho, n, 500_000, rng), alpha
    )
    x_ev = x_alpha_extended_vasicek(n, p0, rho, alpha)
    x_v = x_alpha_inf(p0, rho, alpha)
    assert abs(x_ev - x_sim) < abs(x_v - x_sim)


def test_regress_cm_against_inverse_n_recovers_linear_relation():
    """A synthetic cm_n = cm_inf + (1/n)(cm_1 - cm_inf) must be recovered
    exactly by the regression (Table 4)."""
    cm1_true, cm_inf_true = 7.0, 3.5
    n_values = np.array([2, 5, 10, 50, 100, 500, 1000], dtype=float)
    cm_values = cm_inf_true + (1.0 / n_values) * (cm1_true - cm_inf_true)
    cm1_hat, cm_inf_hat = regress_cm_against_inverse_n(n_values, cm_values)
    assert cm1_hat == pytest.approx(cm1_true, abs=1e-8)
    assert cm_inf_hat == pytest.approx(cm_inf_true, abs=1e-8)


def test_cm_sim_consistent_with_x_alpha_definition():
    p0, n, alpha = 0.02, 50, 0.999
    rho = rho_from_pd(p0)
    sigma_xn = np.sqrt(variance_finite(n, p0, rho))
    rng = np.random.default_rng(3)
    from data.generate_synthetic import simulate_fraction_defaults

    samples = simulate_fraction_defaults(p0, rho, n, 200_000, rng)
    cm = cm_sim(samples, p0, alpha, sigma_xn)
    x_alpha_sim = harrell_davis_quantile(samples, alpha)
    assert p0 + cm * sigma_xn == pytest.approx(x_alpha_sim, rel=1e-8)
