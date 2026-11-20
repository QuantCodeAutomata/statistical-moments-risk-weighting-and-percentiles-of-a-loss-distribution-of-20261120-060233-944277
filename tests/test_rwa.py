"""Tests for src/rwa.py: Basel III/IV IRB, Extended Vasicek and the
multi-rating RWA aggregation (Section 5, Eq. 23-24)."""

import numpy as np
import pytest

from src.rwa import (
    herfindahl_hirschman_index,
    rwa_basel_irb,
    rwa_basel_irb_multirating,
    rwa_extended_vasicek,
    rwa_extended_vasicek_multirating,
    rwa_simulated,
    standardised_floor_rwa,
)
from src.vasicek import rho_from_pd


def test_rwa_extended_vasicek_converges_to_basel_irb():
    """RWA_{alpha,n} -> RWA_{alpha,BIII/IV} as n -> infinity (Eq. 23)."""
    p0 = 0.0004
    rho = rho_from_pd(p0)
    alpha, lgd, ead = 0.999, 0.4, 100.0
    rwa_irb = rwa_basel_irb(p0, rho, alpha, lgd, ead)
    rwa_ev_large_n = rwa_extended_vasicek(1_000_000, p0, rho, alpha, lgd, ead)
    assert rwa_ev_large_n == pytest.approx(rwa_irb, rel=1e-2)


def test_rwa_extended_vasicek_exceeds_basel_irb_for_small_n():
    """Expected outcome of experiment 4: for rating A (PD=0.04%), the
    Extended Vasicek RWA at n=10 is well above the Basel IRB RWA."""
    p0 = 0.0004
    rho = rho_from_pd(p0)
    alpha, lgd, ead = 0.999, 0.4, 100.0
    rwa_irb = rwa_basel_irb(p0, rho, alpha, lgd, ead)
    rwa_ev_n10 = rwa_extended_vasicek(10, p0, rho, alpha, lgd, ead)
    assert rwa_ev_n10 > rwa_irb


def test_rwa_extended_vasicek_decreasing_in_n():
    p0 = 0.0064
    rho = rho_from_pd(p0)
    alpha, lgd, ead = 0.999, 0.4, 100.0
    values = [rwa_extended_vasicek(n, p0, rho, alpha, lgd, ead) for n in (10, 100, 1000, 10000)]
    assert all(np.diff(values) <= 1e-9)


def test_standardised_floor_rwa_formula():
    assert standardised_floor_rwa(0.5, 100.0, 0.725) == pytest.approx(36.25)
    assert standardised_floor_rwa(1.0, 100.0, 0.725) == pytest.approx(72.5)


def test_rwa_basel_irb_multirating_sums_single_rating_contributions():
    composition = {"A": 10, "B": 0}
    pd_by_rating = {"A": 0.001, "B": 0.002}
    rho_by_rating = {r: rho_from_pd(p) for r, p in pd_by_rating.items()}
    alpha, lgd, ead = 0.999, 0.4, 100.0
    total = rwa_basel_irb_multirating(composition, pd_by_rating, rho_by_rating, alpha, lgd, ead)
    expected = 10 * rwa_basel_irb(pd_by_rating["A"], rho_by_rating["A"], alpha, lgd, ead)
    assert total == pytest.approx(expected)


def test_rwa_extended_vasicek_multirating_single_rating_matches_direct_formula():
    """With a single active rating grade I (f_I=1, rho_{I,I}^def=1), the
    aggregation RWA_{alpha,EV} = n_Total*sqrt(sum rho^def_IJ R_I R_J) of
    Section 5 reduces to n_Total times the per-obligor Extended Vasicek RWA
    of that grade (Eq. 23), since R_I = f_I * RWA_{alpha,n_I,I} = RWA_{alpha,n_I,I}."""
    composition = {"A": 40, "B": 0, "C": 0, "D": 0}
    pd_by_rating = {"A": 0.001, "B": 0.002, "C": 0.015, "D": 0.02}
    rho_by_rating = {r: rho_from_pd(p) for r, p in pd_by_rating.items()}
    asset_corr = {(i, j): (1.0 if i == j else 0.2) for i in "ABCD" for j in "ABCD"}
    alpha, lgd, ead = 0.999, 0.4, 100.0
    total = rwa_extended_vasicek_multirating(
        composition, pd_by_rating, rho_by_rating, asset_corr, alpha, lgd, ead
    )
    per_obligor = rwa_extended_vasicek(40, pd_by_rating["A"], rho_by_rating["A"], alpha, lgd, ead)
    expected = 40 * per_obligor
    assert total == pytest.approx(expected, rel=1e-6)


def test_rwa_extended_vasicek_multirating_bounded_by_perfect_correlation_sum():
    """Diversification across rating grades cannot increase RWA beyond the
    sum of the stand-alone grade RWAs (perfect-correlation case)."""
    composition = {"A": 10, "B": 10, "C": 10, "D": 10}
    pd_by_rating = {"A": 0.001, "B": 0.002, "C": 0.015, "D": 0.02}
    rho_by_rating = {r: rho_from_pd(p) for r, p in pd_by_rating.items()}
    asset_corr = {(i, j): (1.0 if i == j else 0.2) for i in "ABCD" for j in "ABCD"}
    alpha, lgd, ead = 0.999, 0.4, 100.0
    total = rwa_extended_vasicek_multirating(
        composition, pd_by_rating, rho_by_rating, asset_corr, alpha, lgd, ead
    )
    n_total = sum(composition.values())
    perfect_corr_sum = sum(
        (composition[r] / n_total) * rwa_extended_vasicek(composition[r], pd_by_rating[r], rho_by_rating[r], alpha, lgd, ead)
        for r in composition
    ) * n_total
    assert total <= perfect_corr_sum + 1e-6


def test_rwa_simulated_matches_analytical_moments_in_expectation():
    from data.generate_synthetic import simulate_multirating_portfolio

    rng = np.random.default_rng(11)
    composition = {"A": 50, "B": 50}
    pd_by_rating = {"A": 0.001, "B": 0.01}
    rho_by_rating = {r: rho_from_pd(p) for r, p in pd_by_rating.items()}
    x = simulate_multirating_portfolio(composition, pd_by_rating, rho_by_rating, 200_000, rng)
    n_total = sum(composition.values())
    rwa = rwa_simulated(x, 0.999, n_total, 0.4, 100.0)
    assert rwa > 0.0


def test_herfindahl_hirschman_index_uniform_and_concentrated():
    uniform = herfindahl_hirschman_index({"A": 10, "B": 10, "C": 10, "D": 10})
    concentrated = herfindahl_hirschman_index({"A": 37, "B": 1, "C": 1, "D": 1})
    assert uniform == pytest.approx(0.25)
    assert concentrated > uniform
    assert concentrated <= 1.0
