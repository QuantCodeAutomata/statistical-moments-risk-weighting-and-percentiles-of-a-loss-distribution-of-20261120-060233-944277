"""Synthetic loss-fraction data for the experiments: Monte Carlo simulation of
the single-factor homogeneous credit portfolio model (Section 2, Eq. 1-4;
Section 4.1 simulation set-up).

The paper draws, for every simulation run, one systematic factor f and n
idiosyncratic shocks epsilon_i, correlates them via Cholesky decomposition of
the homogeneous correlation matrix C_n, and counts defaults where
z_i = sqrt(rho) f + sqrt(1-rho) epsilon_i <= tau. Conditional on f, the
z_i are i.i.d. so the number of defaults d is exactly distributed as
Binomial(n, p(f)) with p(f) = Phi[(tau - sqrt(rho) f) / sqrt(1 - rho)]
(Eq. 3). Drawing d directly from this conditional binomial distribution
reproduces the identical distribution of x = d/n as simulating every
epsilon_i individually, while keeping memory use independent of n (critical
for n up to 2000 and 500,000 runs); numpy's vectorised binomial sampler is
used instead of materialising n correlated normals per run.
"""

from __future__ import annotations

import numpy as np

from src.vasicek import tau_from_pd


def conditional_default_probability(f: np.ndarray, tau: float, rho: float) -> np.ndarray:
    """p(f) = Phi[(tau - sqrt(rho) f) / sqrt(1 - rho)] (Eq. 3)."""
    from scipy.stats import norm

    return norm.cdf((tau - np.sqrt(rho) * f) / np.sqrt(1.0 - rho))


def simulate_fraction_defaults(
    p0: float, rho: float, n: int, n_sims: int, rng: np.random.Generator
) -> np.ndarray:
    """Simulate n_sims draws of the loss fraction x = d/n for a homogeneous
    portfolio of n obligors with unconditional default probability p0 and
    asset correlation rho (Section 4.1, steps 1-4)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    tau = tau_from_pd(p0)
    f = rng.standard_normal(n_sims)
    p_f = conditional_default_probability(f, tau, rho)
    d = rng.binomial(n, p_f)
    return d / n


def simulate_multirating_portfolio(
    composition: dict[str, int],
    pd_by_rating: dict[str, float],
    rho_by_rating: dict[str, float],
    n_sims: int,
    rng: np.random.Generator,
    sector_correlation: dict[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """Simulate n_sims draws of the total loss fraction for a portfolio with
    several rating grades (Section 5), sharing one systematic factor f per
    run across all rating grades (single-factor model, Section 2); each
    rating grade uses its own within-rating correlation rho_I from Eq. (22).
    Returns the fraction of defaulted obligors in the whole portfolio,
    x = (sum_I d_I) / n_Total.

    An optional ``sector_correlation`` draws a separate, correlated factor
    f_I per rating grade instead (via a Cholesky decomposition of that
    matrix), which was tested against the asset-correlation matrix of
    Table 5 and found to reproduce Table 6 markedly worse than the default
    shared-factor model (see results/RESULTS.md, "Deviations from the
    paper"); it is kept only for that comparison and is not used by the
    experiment.
    """
    n_total = sum(composition.values())
    if n_total < 1:
        raise ValueError("portfolio must contain at least one obligor")
    ratings = [r for r in composition if composition[r] > 0]
    if sector_correlation is None:
        f_global = rng.standard_normal(n_sims)
        f_by_rating = {r: f_global for r in ratings}
    else:
        corr = np.array(
            [[sector_correlation[(i, j)] for j in ratings] for i in ratings]
        )
        chol = np.linalg.cholesky(corr)
        z = rng.standard_normal((n_sims, len(ratings)))
        f = z @ chol.T
        f_by_rating = {r: f[:, idx] for idx, r in enumerate(ratings)}

    total_defaults = np.zeros(n_sims, dtype=np.int64)
    for rating in ratings:
        n_i = composition[rating]
        tau_i = tau_from_pd(pd_by_rating[rating])
        p_f_i = conditional_default_probability(f_by_rating[rating], tau_i, rho_by_rating[rating])
        total_defaults += rng.binomial(n_i, p_f_i)
    return total_defaults / n_total
