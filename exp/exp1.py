"""Experiment 1: verification of the central moments (standard deviation,
3rd and 4th central moments) of the loss fraction x = d/n of a finite
homogeneous credit portfolio against Monte Carlo simulation (Section 3.1,
Theorem 2, Eq. 10-11; Section 4.1-4.2, Figures 1-3).

Run: python -m exp.exp1 [--full] [--quick]
"""

from __future__ import annotations

import argparse
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
    N_GRID_EXP1,
    N_SIMS,
    N_SIMS_RUN,
    P0_GRID_EXP1,
    RANDOM_SEED,
)
from src.moments import central_moment_finite, variance_finite, vasicek_central_moment
from src.vasicek import rho_from_pd, sigma_x_inf2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_configuration(p0: float, n: int, n_sims: int, rng: np.random.Generator) -> dict:
    """Simulate one (P0, n) configuration and compute theoretical moments
    (Section 4.1, steps 1-4)."""
    rho = rho_from_pd(p0)
    x = simulate_fraction_defaults(p0, rho, n, n_sims, rng)

    sigma_sim = float(np.std(x, ddof=1))
    mu3_sim = float(np.mean((x - p0) ** 3))
    mu4_sim = float(np.mean((x - p0) ** 4))

    sigma_theory = np.sqrt(variance_finite(n, p0, rho))
    mu3_theory = central_moment_finite(3, n, p0, rho)
    mu4_theory = central_moment_finite(4, n, p0, rho)

    sigma_inf = np.sqrt(sigma_x_inf2(p0, rho))
    mu3_inf = vasicek_central_moment(3, p0, rho)
    mu4_inf = vasicek_central_moment(4, p0, rho)

    return {
        "P0": p0,
        "rho": rho,
        "n": n,
        "sigma_sim": sigma_sim,
        "sigma_theory": sigma_theory,
        "sigma_inf": sigma_inf,
        "mu3_sim": mu3_sim,
        "mu3_theory": mu3_theory,
        "mu3_inf": mu3_inf,
        "mu4_sim": mu4_sim,
        "mu4_theory": mu4_theory,
        "mu4_inf": mu4_inf,
    }


def run_experiment(n_sims: int, n_grid: list[int], p0_grid: list[float]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    t0 = time.time()
    for p0 in p0_grid:
        for n in n_grid:
            rows.append(run_configuration(p0, n, n_sims, rng))
    logger.info(
        "exp1: %d configurations (P0 x n = %d x %d), n_sims=%d, %.1fs",
        len(rows), len(p0_grid), len(n_grid), n_sims, time.time() - t0,
    )
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, out_path: Path) -> None:
    """Central moments as a function of log(n), one panel per P0
    (Section 4.2, Figures 1-3)."""
    p0_values = sorted(df["P0"].unique())
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    moment_specs = [
        ("sigma_sim", "sigma_theory", "Standard deviation sigma_x,n", axes[0]),
        ("mu3_sim", "mu3_theory", "3rd central moment mu_3,x,n", axes[1]),
        ("mu4_sim", "mu4_theory", "4th central moment mu_4,x,n", axes[2]),
    ]
    cmap = plt.get_cmap("viridis", len(p0_values))
    for idx, p0 in enumerate(p0_values):
        sub = df[df["P0"] == p0].sort_values("n")
        color = cmap(idx)
        for sim_col, theory_col, title, ax in moment_specs:
            ax.plot(np.log(sub["n"]), sub[sim_col], "o", color=color, markersize=3,
                     label=f"P0={p0} sim" if ax is axes[0] else None)
            ax.plot(np.log(sub["n"]), sub[theory_col], "-", color=color,
                     label=f"P0={p0} theory" if ax is axes[0] else None)
    for _, _, title, ax in moment_specs:
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("log(n)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Theorem 2: finite-portfolio central moments vs. simulation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="Use the full paper grid (default; kept for interface symmetry).")
    parser.add_argument("--quick", action="store_true",
                         help="Use N_SIMS_RUN instead of the paper's N_SIMS, for fast smoke tests.")
    args = parser.parse_args()

    n_sims = N_SIMS_RUN if args.quick else N_SIMS
    df = run_experiment(n_sims, N_GRID_EXP1, P0_GRID_EXP1)

    df["sigma_abs_rel_err"] = np.abs(df["sigma_sim"] - df["sigma_theory"]) / df["sigma_theory"]
    df["mu3_abs_err"] = np.abs(df["mu3_sim"] - df["mu3_theory"])
    df["mu4_abs_err"] = np.abs(df["mu4_sim"] - df["mu4_theory"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "exp1_central_moments.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s (%d rows)", csv_path, len(df))

    plot_path = RESULTS_DIR / "exp1_moments_vs_logn.png"
    make_plots(df, plot_path)
    logger.info("Saved %s", plot_path)

    summary = {
        "n_sims": n_sims,
        "n_configurations": len(df),
        "median_sigma_rel_err": float(df["sigma_abs_rel_err"].median()),
        "max_sigma_rel_err_n_ge_30": float(
            df.loc[df["n"] >= 30, "sigma_abs_rel_err"].max()
        ),
        "n_grid": N_GRID_EXP1,
        "p0_grid": P0_GRID_EXP1,
    }
    pd.Series(summary).to_json(RESULTS_DIR / "exp1_summary.json", indent=2)
    logger.info("exp1 summary: %s", summary)


if __name__ == "__main__":
    main()
