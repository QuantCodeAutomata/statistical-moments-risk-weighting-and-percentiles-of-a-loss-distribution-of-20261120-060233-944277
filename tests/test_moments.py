"""Tests for src/moments.py: central moments of the finite-portfolio loss
fraction (Section 3.1, Theorem 2)."""

import numpy as np
import pytest

from src.moments import (
    a_ml,
    central_moment_finite,
    one_obligor_central_moment,
    variance_finite,
    vasicek_central_moment,
)
from src.vasicek import rho_from_pd, sigma_x_inf2


def test_one_obligor_central_moment_matches_bernoulli_formulas():
    p0 = 0.03
    assert one_obligor_central_moment(2, p0) == pytest.approx(p0 * (1 - p0))
    assert one_obligor_central_moment(3, p0) == pytest.approx(p0 * (1 - p0) * (1 - 2 * p0))
    assert one_obligor_central_moment(4, p0) == pytest.approx(
        p0 * (1 - p0) * (1 - 3 * p0 + 3 * p0 ** 2)
    )


def test_a_ml_table2_values_for_m_equals_3():
    """Table 2: a_3,1 = -1, a_3,2 = 3, a_3,3 = -2, independent of n and P0."""
    for n in (5, 50, 500):
        for p0 in (0.001, 0.02, 0.1):
            assert a_ml(3, 1, n, p0) == pytest.approx(-1.0)
            assert a_ml(3, 2, n, p0) == pytest.approx(3.0)
            assert a_ml(3, 3, n, p0) == pytest.approx(-2.0)


def test_central_moment_finite_reduces_to_one_obligor_at_n_equals_1():
    p0, rho = 0.02, 0.164
    for m in (2, 3, 4):
        assert central_moment_finite(m, 1, p0, rho) == pytest.approx(
            one_obligor_central_moment(m, p0)
        )


def test_central_moment_finite_variance_matches_eq10():
    """The m=2 central moment from Theorem 2 must agree with the direct
    variance formula of Eq. 10."""
    p0, rho = 0.01, 0.15
    for n in (2, 10, 100):
        mu2 = central_moment_finite(2, n, p0, rho)
        var_eq10 = variance_finite(n, p0, rho)
        assert mu2 == pytest.approx(var_eq10, rel=1e-8)


def test_central_moment_finite_converges_to_vasicek_limit():
    """As n -> infinity, (mu_m)_{x,n} -> (mu_m)_{x,inf} (Theorem 2)."""
    p0 = 0.02
    rho = rho_from_pd(p0)
    for m in (2, 3, 4):
        mu_n_small = central_moment_finite(m, 5, p0, rho)
        mu_n_large = central_moment_finite(m, 5000, p0, rho)
        mu_inf = vasicek_central_moment(m, p0, rho)
        assert abs(mu_n_large - mu_inf) < abs(mu_n_small - mu_inf)
        assert mu_n_large == pytest.approx(mu_inf, rel=2e-2)


def test_variance_finite_decreasing_towards_vasicek_limit():
    """sigma_x,n^2 strictly decreases in n and is bounded below by the
    Vasicek-limit variance (Eq. 10)."""
    p0, rho = 0.02, 0.164
    sigma_inf2 = sigma_x_inf2(p0, rho)
    variances = [variance_finite(n, p0, rho) for n in (2, 5, 20, 100, 1000)]
    assert all(np.diff(variances) < 0)
    assert variances[-1] > sigma_inf2


def test_variance_finite_matches_small_n_expected_outcome():
    """Expected outcome of experiment 1 (Section 4.2): for small n the finite
    portfolio standard deviation is much larger than the Vasicek-limit
    standard deviation sigma_x,inf, and Eq. 10 exactly reduces to the
    idiosyncratic (binomial) term plus the (1-1/n) weighted Vasicek-limit
    variance, so it can be reconstructed from its two independently
    identifiable pieces."""
    p0, rho, n = 0.02, 0.164, 3
    sigma_inf2 = sigma_x_inf2(p0, rho)
    sigma_n3 = np.sqrt(variance_finite(n, p0, rho))
    sigma_inf = np.sqrt(sigma_inf2)
    eq10_variance = p0 * (1.0 - p0) / n + (1.0 - 1.0 / n) * sigma_inf2
    assert variance_finite(n, p0, rho) == pytest.approx(eq10_variance, rel=1e-9)
    assert sigma_n3 > 2 * sigma_inf


def test_central_moment_finite_rejects_invalid_n():
    with pytest.raises(ValueError):
        central_moment_finite(2, 0, 0.01, 0.1)
