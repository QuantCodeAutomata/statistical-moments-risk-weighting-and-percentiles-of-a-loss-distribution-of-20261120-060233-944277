"""Experiment 2: capital multiple comparison between the Cornish-Fisher
expansion, the 1/n-weighted Extended-Vasicek formula and Monte Carlo
simulation, for a finite homogeneous credit portfolio (Section 3.2 of the
paper).

For each (P0, n) configuration, the capital multiple cm_n is estimated three
ways:

* cm_n,sim: via simulation and the Harrell-Davis quantile estimator.
* cm_n,EV: the 1/n-weighted capital multiple of Eq. (19), built from the
  one-obligor capital multiple cm_1 (Eq. 20) and the Vasicek-limit capital
  multiple cm_inf.
* cm_n,CF: the Cornish-Fisher capital multiple of Eq. (15), with (s, k) fitted
  by minimising the moment-matching objective E of Eq. (17) and checked
  against the admissible domain R of Eq. (18).

Across the n-grid, cm_n,sim is also regressed against 1/n (Section 3.2.2,
Table 4) to recover cm_1,reg and cm_inf,reg, which are compared against the
theoretical cm_1 (Eq. 20) and cm_inf.

Run: python -m exp.exp2 [--full] [--quick]
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
from scipy.stats import norm

from data.generate_synthetic import simulate_fraction_defaults
from src.config import (
    ALPHA,
    N_GRID_EXP2,
    N_SIMS,
    N_SIMS_RUN,
    P0_GRID_EXP2,
    RANDOM_SEED,
)
from src.cornish_fisher import admissible_domain, cm_cf, fit_sk
from src.moments import central_moment_finite, variance_finite
from src.percentiles import (
    cm_ev,
    cm_one_obligor,
    cm_sim,
    cm_vasicek_limit,
    regress_cm_against_inverse_n,
)
from src.vasicek import rho_from_pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_configuration(p0: float, n: int, n_sims: int, rng: np.random.Generator) -> dict:
    """Simulate one (P0, n) configuration and compute the three capital
    multiple estimates (Section 3.2, Eq. 15, 17, 18, 19, 20).

    Args:
        p0: unconditional default probability.
        n: number of obligors in the homogeneous portfolio.
        n_sims: number of Monte Carlo runs used to estimate cm_n,sim.
        rng: shared random generator, reused across the whole grid.

    Returns:
        A dict with one row of results: P0, rho, n, the three capital
        multiples, the fitted Cornish-Fisher parameters (s, k, E) and whether
        (s, k) lie in the admissible domain R.
    """
    rho = rho_from_pd(p0)
    samples = simulate_fraction_defaults(p0, rho, n, n_sims, rng)
    sigma_x_n = np.sqrt(variance_finite(n, p0, rho))

    cm_n_sim = cm_sim(samples, p0, ALPHA, sigma_x_n)
    cm_n_ev = cm_ev(n, p0, rho, ALPHA)

    mu3_target = central_moment_finite(3, n, p0, rho)
    mu4_target = central_moment_finite(4, n, p0, rho)
    s, k, e = fit_sk(mu3_target, mu4_target, sigma_x_n)
    u_alpha = norm.ppf(ALPHA)
    cm_n_cf = cm_cf(u_alpha, s, k)
    in_domain_r = admissible_domain(s, k)

    return {
        "P0": p0,
        "rho": rho,
        "n": n,
        "cm_n_sim": cm_n_sim,
        "cm_n_ev": cm_n_ev,
        "cm_n_cf": cm_n_cf,
        "s": s,
        "k": k,
        "E": e,
        "in_domain_R": in_domain_r,
    }


def run_experiment(n_sims: int, n_grid: list[int], p0_grid: list[float]) -> pd.DataFrame:
    """Run every (P0, n) configuration of the grid with one shared, seeded
    random generator (Section 4.1 simulation set-up).

    Args:
        n_sims: number of Monte Carlo runs per configuration.
        n_grid: grid of obligor counts.
        p0_grid: grid of unconditional default probabilities.

    Returns:
        One row per (P0, n) configuration (see run_configuration).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    t0 = time.time()
    for p0 in p0_grid:
        for n in n_grid:
            rows.append(run_configuration(p0, n, n_sims, rng))
    logger.info(
        "exp2: %d configurations (P0 x n = %d x %d), n_sims=%d, %.1fs",
        len(rows), len(p0_grid), len(n_grid), n_sims, time.time() - t0,
    )
    return pd.DataFrame(rows)


def build_table4(df: pd.DataFrame, p0_grid: list[float]) -> pd.DataFrame:
    """Reproduce Table 4: for each P0, regress cm_n,sim against 1/n over the
    full n-grid and compare against the theoretical cm_1 and cm_inf
    (Section 3.2.2, Eq. 20, Eq. 8).

    Args:
        df: the full per-(P0, n) table produced by run_experiment.
        p0_grid: grid of unconditional default probabilities.

    Returns:
        One row per P0: P0, rho, cm_1, cm_inf, cm_1_reg, cm_inf_reg.
    """
    rows = []
    for p0 in p0_grid:
        sub = df[df["P0"] == p0].sort_values("n")
        rho = float(sub["rho"].iloc[0])
        cm_1_reg, cm_inf_reg = regress_cm_against_inverse_n(
            sub["n"].to_numpy(), sub["cm_n_sim"].to_numpy()
        )
        rows.append(
            {
                "P0": p0,
                "rho": rho,
                "cm_1": float(cm_one_obligor(p0)),
                "cm_inf": float(cm_vasicek_limit(p0, rho, ALPHA)),
                "cm_1_reg": cm_1_reg,
                "cm_inf_reg": cm_inf_reg,
            }
        )
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, out_path: Path) -> None:
    """Capital multiple as a function of n (log scale), one panel per P0
    (Section 3.2, Figure 5): cm_n,sim as markers, cm_n,EV as a solid line,
    cm_n,CF as a dashed line.

    Args:
        df: the full per-(P0, n) table produced by run_experiment.
        out_path: path the figure is saved to.
    """
    p0_values = sorted(df["P0"].unique())
    fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True)
    for ax, p0 in zip(axes.flatten(), p0_values):
        sub = df[df["P0"] == p0].sort_values("n")
        ax.plot(sub["n"], sub["cm_n_sim"], "o", markersize=3, label="cm_n,sim")
        ax.plot(sub["n"], sub["cm_n_ev"], "-", label="cm_n,EV")
        ax.plot(sub["n"], sub["cm_n_cf"], "--", label="cm_n,CF")
        ax.set_xscale("log")
        ax.set_title(f"P0 = {p0}")
        ax.set_xlabel("n (log scale)")
        ax.set_ylabel("capital multiple")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Experiment 2: capital multiple, Cornish-Fisher vs weighting vs simulation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="Use the full paper grid (default; kept for interface symmetry).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use N_SIMS_RUN instead of the paper's N_SIMS, for fast smoke tests.",
    )
    args = parser.parse_args()

    n_sims = N_SIMS_RUN if args.quick else N_SIMS
    t0 = time.time()

    df = run_experiment(n_sims, N_GRID_EXP2, P0_GRID_EXP2)
    table4 = build_table4(df, P0_GRID_EXP2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "exp2_capital_multiple.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s (%d rows)", csv_path, len(df))

    table4_path = RESULTS_DIR / "exp2_table4_regression.csv"
    table4.to_csv(table4_path, index=False)
    logger.info("Saved %s (%d rows)", table4_path, len(table4))

    plot_path = RESULTS_DIR / "exp2_capital_multiple.png"
    make_plots(df, plot_path)
    logger.info("Saved %s", plot_path)

    summary = {
        "n_sims": n_sims,
        "n_configurations": len(df),
        "frac_out_of_domain_R": float((~df["in_domain_R"]).mean()),
        "table4": table4.to_dict(orient="records"),
    }
    with open(RESULTS_DIR / "exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("exp2: total elapsed %.1fs", time.time() - t0)
    logger.info("exp2 summary: %s", summary)


if __name__ == "__main__":
    main()
