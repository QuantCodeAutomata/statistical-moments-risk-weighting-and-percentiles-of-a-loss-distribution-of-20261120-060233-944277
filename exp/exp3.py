"""Experiment 3: 99.9% percentile comparison -- Extended Vasicek vs Vasicek
vs Cornish-Fisher vs Monte Carlo simulation, for a finite homogeneous credit
portfolio (Section 3.2.2 of van der Burgt, "Statistical Moments, Risk
Weighting and Percentiles of a Loss Distribution of a Finite Homogeneous
Credit Portfolio; extending the Vasicek Distribution").

For every (P0, n) configuration this script compares four estimates of the
99.9% percentile x_{0.999} of the loss fraction x = d/n:

* x_sim: the Harrell-Davis quantile estimator applied to Monte Carlo samples
  (Section 4.1).
* x_ev:  the Extended Vasicek percentile x_{alpha,n} (Eq. 21), which blends
  the one-obligor and the Vasicek-limit percentile with 1/n weights and
  therefore depends on n.
* x_v:   the plain (original) Vasicek-limit percentile x_{alpha,inf}
  (Eq. 8), which does not depend on n and is expected to be a poor
  approximation for small n.
* x_cf:  the Cornish-Fisher percentile (Eq. 13 + 15), obtained by fitting the
  skewness/kurtosis parameters (s, k) of the Cornish-Fisher expansion to the
  finite-portfolio 3rd and 4th central moments (Theorem 2, Eq. 11) via
  BFGS (Eq. 17).

The Vasicek-limit variance (Eq. 7) underlying x_v is not used directly here;
it is implicitly folded into x_alpha_inf (Eq. 8) and into the Extended
Vasicek percentile (Eq. 21) via src.percentiles.cm_vasicek_limit.

Run: python -m exp.exp3 [--full] [--quick]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.generate_synthetic import simulate_fraction_defaults
from src.config import (
    ALPHA,
    N_GRID_EXP3,
    N_SIMS,
    N_SIMS_RUN,
    P0_GRID_EXP3,
    RANDOM_SEED,
)
from src.cornish_fisher import fit_sk, x_alpha_cf
from src.moments import central_moment_finite, variance_finite
from src.percentiles import harrell_davis_quantile, x_alpha_extended_vasicek
from src.vasicek import rho_from_pd, x_alpha_inf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_configuration(p0: float, n: int, n_sims: int, rng: np.random.Generator) -> dict:
    """Simulate one (P0, n) configuration and compute the four percentile
    estimates (Section 3.2.2, Eq. 7, 8, 21; Section 3.2.1, Eq. 13-17).

    Parameters
    ----------
    p0 : float
        Unconditional default probability of the configuration.
    n : int
        Number of obligors in the (homogeneous) portfolio.
    n_sims : int
        Number of Monte Carlo simulation runs used for x_sim.
    rng : np.random.Generator
        Shared random generator, reused across the whole (P0, n) grid.

    Returns
    -------
    dict
        One row of results, keyed by P0, rho, n, x_sim, x_ev, x_v, x_cf,
        x_cf_clipped, s, k, E.
    """
    rho = rho_from_pd(p0)
    samples = simulate_fraction_defaults(p0, rho, n, n_sims, rng)

    x_sim = harrell_davis_quantile(samples, ALPHA)
    x_ev = x_alpha_extended_vasicek(n, p0, rho, ALPHA)
    x_v = x_alpha_inf(p0, rho, ALPHA)

    sigma_x_n = float(np.sqrt(variance_finite(n, p0, rho)))
    mu3_target = central_moment_finite(3, n, p0, rho)
    mu4_target = central_moment_finite(4, n, p0, rho)
    s, k, e = fit_sk(mu3_target, mu4_target, sigma_x_n)
    x_cf = x_alpha_cf(p0, ALPHA, s, k, sigma_x_n)
    x_cf_clipped = max(x_cf, 0.0)

    return {
        "P0": p0,
        "rho": rho,
        "n": n,
        "x_sim": x_sim,
        "x_ev": x_ev,
        "x_v": x_v,
        "x_cf": x_cf,
        "x_cf_clipped": x_cf_clipped,
        "s": s,
        "k": k,
        "E": e,
    }


def run_experiment(n_sims: int, n_grid: list[int], p0_grid: list[float]) -> pd.DataFrame:
    """Run the full (P0, n) grid, sharing one random generator seeded from
    src.config.RANDOM_SEED across the whole grid (Section 4.1)."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    t0 = time.time()
    for p0 in p0_grid:
        for n in n_grid:
            rows.append(run_configuration(p0, n, n_sims, rng))
    logger.info(
        "exp3: %d configurations (P0 x n = %d x %d), n_sims=%d, %.1fs",
        len(rows), len(p0_grid), len(n_grid), n_sims, time.time() - t0,
    )
    return pd.DataFrame(rows)


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add absolute and relative error columns of x_ev, x_v, x_cf against
    x_sim (relative errors only where x_sim > 0, Section 3.2.2)."""
    df = df.copy()
    df["ev_abs_err"] = (df["x_ev"] - df["x_sim"]).abs()
    df["v_abs_err"] = (df["x_v"] - df["x_sim"]).abs()
    df["cf_abs_err"] = (df["x_cf"] - df["x_sim"]).abs()

    sim_positive = df["x_sim"] > 0
    df["ev_rel_err"] = np.where(sim_positive, df["ev_abs_err"] / df["x_sim"], np.nan)
    df["v_rel_err"] = np.where(sim_positive, df["v_abs_err"] / df["x_sim"], np.nan)
    df["cf_rel_err"] = np.where(sim_positive, df["cf_abs_err"] / df["x_sim"], np.nan)
    return df


def make_plots(df: pd.DataFrame, out_path: Path) -> None:
    """One subplot per P0 (4x2 grid), x_sim (markers), x_ev (line), x_v
    (flat line), x_cf (dashed line) against n on a log scale (Figure 6)."""
    p0_values = sorted(df["P0"].unique())
    n_rows, n_cols = 4, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 16))
    axes_flat = axes.flatten()

    for ax, p0 in zip(axes_flat, p0_values):
        sub = df[df["P0"] == p0].sort_values("n")
        ax.plot(sub["n"], sub["x_sim"], "o", color="black", markersize=4, label="x_sim")
        ax.plot(sub["n"], sub["x_ev"], "-", color="tab:blue", label="x_ev (Extended Vasicek)")
        ax.plot(sub["n"], sub["x_v"], "-", color="tab:red", label="x_v (Vasicek limit)")
        ax.plot(sub["n"], sub["x_cf"], "--", color="tab:green", label="x_cf (Cornish-Fisher)")
        ax.set_xscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("99.9% percentile of loss fraction x")
        ax.set_title(f"P0 = {p0}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for ax in axes_flat[len(p0_values):]:
        ax.set_visible(False)

    fig.suptitle("Figure 6: 99.9% percentile vs n -- Extended Vasicek, Vasicek, Cornish-Fisher, simulation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def build_summary(df: pd.DataFrame, n_sims: int) -> dict:
    """Summary of the error distributions over the whole grid and restricted
    to the small-n regime (n <= 10), where Extended Vasicek is expected to
    most improve on the plain Vasicek-limit percentile (Section 3.2.2)."""
    small_n = df[df["n"] <= 10]

    def stats(sub: pd.DataFrame) -> dict:
        return {
            "ev_abs_err_median": float(sub["ev_abs_err"].median()),
            "ev_abs_err_max": float(sub["ev_abs_err"].max()),
            "v_abs_err_median": float(sub["v_abs_err"].median()),
            "v_abs_err_max": float(sub["v_abs_err"].max()),
            "cf_abs_err_median": float(sub["cf_abs_err"].median()),
            "cf_abs_err_max": float(sub["cf_abs_err"].max()),
        }

    return {
        "n_sims": n_sims,
        "n_configurations": len(df),
        "all_n": stats(df),
        "small_n_le_10": stats(small_n),
        "n_configurations_small_n": len(small_n),
        "n_grid": N_GRID_EXP3,
        "p0_grid": P0_GRID_EXP3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="Use the full paper grid (default; kept for interface symmetry).")
    parser.add_argument("--quick", action="store_true",
                         help="Use N_SIMS_RUN instead of the paper's N_SIMS, for fast smoke tests.")
    args = parser.parse_args()

    n_sims = N_SIMS_RUN if args.quick else N_SIMS
    df = run_experiment(n_sims, N_GRID_EXP3, P0_GRID_EXP3)
    df = add_error_columns(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "exp3_percentiles.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s (%d rows)", csv_path, len(df))

    plot_path = RESULTS_DIR / "exp3_percentiles.png"
    make_plots(df, plot_path)
    logger.info("Saved %s", plot_path)

    summary = build_summary(df, n_sims)
    summary_path = RESULTS_DIR / "exp3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("exp3 summary: %s", summary)


if __name__ == "__main__":
    main()
