"""Tests for data/generate_synthetic.py: the single-factor Monte Carlo
sampler used by every experiment (Section 4.1)."""

import numpy as np
import pytest

from data.generate_synthetic import (
    conditional_default_probability,
    simulate_fraction_defaults,
    simulate_multirating_portfolio,
)
from src.moments import central_moment_finite, variance_finite
from src.vasicek import rho_from_pd, tau_from_pd


def test_conditional_default_probability_is_a_probability():
    tau = tau_from_pd(0.02)
    f = np.linspace(-4, 4, 50)
    p = conditional_default_probability(f, tau, 0.2)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all(np.diff(p) <= 0.0)  # decreasing in f (Eq. 3)


def test_simulate_fraction_defaults_matches_theoretical_moments():
    """The simulated mean, variance and 3rd moment of x must match the
    theoretical values (Eq. 10, Theorem 2) within Monte Carlo error."""
    rng = np.random.default_rng(42)
    p0, n, n_sims = 0.02, 20, 200_000
    rho = rho_from_pd(p0)
    x = simulate_fraction_defaults(p0, rho, n, n_sims, rng)
    assert x.shape == (n_sims,)
    assert np.all((x >= 0.0) & (x <= 1.0))
    assert np.mean(x) == pytest.approx(p0, abs=5e-4)
    sim_var = np.var(x, ddof=1)
    th_var = variance_finite(n, p0, rho)
    assert sim_var == pytest.approx(th_var, rel=0.1)
    sim_mu3 = np.mean((x - p0) ** 3)
    th_mu3 = central_moment_finite(3, n, p0, rho)
    assert sim_mu3 == pytest.approx(th_mu3, rel=0.25)


def test_simulate_fraction_defaults_rejects_invalid_input():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_fraction_defaults(0.01, 0.1, 0, 1000, rng)
    with pytest.raises(ValueError):
        simulate_fraction_defaults(0.01, 0.1, 10, 0, rng)


def test_simulate_multirating_portfolio_fraction_bounds_and_mean():
    rng = np.random.default_rng(7)
    composition = {"A": 10, "B": 10, "C": 10, "D": 10}
    pd_by_rating = {"A": 0.001, "B": 0.002, "C": 0.015, "D": 0.020}
    rho_by_rating = {r: rho_from_pd(p) for r, p in pd_by_rating.items()}
    x = simulate_multirating_portfolio(composition, pd_by_rating, rho_by_rating, 100_000, rng)
    assert np.all((x >= 0.0) & (x <= 1.0))
    expected_mean = sum(n_i * pd_by_rating[r] for r, n_i in composition.items()) / sum(
        composition.values()
    )
    assert np.mean(x) == pytest.approx(expected_mean, abs=2e-3)


def test_simulate_multirating_portfolio_rejects_empty_portfolio():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_multirating_portfolio({"A": 0}, {"A": 0.01}, {"A": 0.1}, 100, rng)
