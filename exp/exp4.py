"""Experiment 4: RWA comparison for single rating grades -- Basel III/IV vs.
Extended Vasicek, against the 72.5% standardised-approach output floor
(Section 5, Figure 7).

For two single-rating-grade portfolios (S&P rating A, PD = 0.04%; S&P rating
BB, PD = 0.64%), this experiment computes, as a function of the number of
obligors n:

* the Basel III/IV IRB RWA (Vasicek limit, independent of n), from the
  Vasicek-limit percentile of Eq. (8);
* the Extended Vasicek RWA (Eq. 23), which depends on n and converges to the
  Basel III/IV figure as n -> infinity;
* the 72.5% standardised-approach output floor RWA (also independent of n),
  using the BCBS correlation of Eq. (22) to obtain rho from PD for both the
  IRB and Extended Vasicek calculations.

The experiment then locates, on the N_GRID_EXP4 grid, the largest n at which
the Extended Vasicek RWA still exceeds the output-floor RWA, reproducing the
paper's claim that the 72.5% floor is "sufficiently prudent" except for very
small portfolios.

Run: python -m exp.exp4 [--full] [--quick]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    ALPHA,
    EAD_EXP4,
    LGD_EXP4,
    N_GRID_EXP4,
    N_TABLE_EXP4,
    OUTPUT_FLOOR,
    RATING_A_PD,
    RATING_A_RISK_WEIGHT_SA,
    RATING_BB_PD,
    RATING_BB_RISK_WEIGHT_SA,
)
from src.rwa import rwa_basel_irb, rwa_extended_vasicek, standardised_floor_rwa
from src.vasicek import rho_from_pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Case name -> (PD, standardised-approach risk weight), Section 5.
CASES: dict[str, tuple[float, float]] = {
    "A": (RATING_A_PD, RATING_A_RISK_WEIGHT_SA),
    "BB": (RATING_BB_PD, RATING_BB_RISK_WEIGHT_SA),
}


def run_case(case: str, p0: float, risk_weight_sa: float, n_grid: list[int]) -> pd.DataFrame:
    """Compute, for one rating case, the Basel III/IV IRB RWA, the
    standardised-approach floor RWA (both constant in n) and the Extended
    Vasicek RWA for every n in n_grid (Section 5, Eq. 8, 22, 23).

    Returns one row per n with columns: case, n, p0, rho, rwa_irb, rwa_ev,
    rwa_floor.
    """
    rho = rho_from_pd(p0)
    rwa_irb = rwa_basel_irb(p0, rho, ALPHA, LGD_EXP4, EAD_EXP4)
    rwa_floor = standardised_floor_rwa(risk_weight_sa, EAD_EXP4, OUTPUT_FLOOR)

    rows = []
    for n in n_grid:
        rwa_ev = rwa_extended_vasicek(n, p0, rho, ALPHA, LGD_EXP4, EAD_EXP4)
        rows.append({
            "case": case,
            "n": n,
            "p0": p0,
            "rho": rho,
            "rwa_irb": rwa_irb,
            "rwa_ev": rwa_ev,
            "rwa_floor": rwa_floor,
        })
    return pd.DataFrame(rows)


def find_floor_binds_below_n(df_case: pd.DataFrame) -> Optional[int]:
    """Largest n (on the grid underlying df_case) at which rwa_ev still
    exceeds rwa_floor.

    Below this n, the Extended Vasicek RWA is higher than the standardised
    floor RWA (i.e. the floor alone would understate the finite-portfolio
    risk); at and above this n, the Extended Vasicek RWA has fallen to or
    below the floor. Returns None if rwa_ev never exceeds rwa_floor on the
    grid.
    """
    above_floor = df_case.loc[df_case["rwa_ev"] > df_case["rwa_floor"], "n"]
    if above_floor.empty:
        return None
    return int(above_floor.max())


def make_plot(df: pd.DataFrame, cases: dict[str, tuple[float, float]], out_path: Path) -> None:
    """Two side-by-side panels (one per case), x = n on a log scale, showing
    the Extended Vasicek RWA curve against the flat Basel III/IV IRB and
    output-floor reference lines (Section 5, Figure 7)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, case in zip(axes, cases):
        sub = df[df["case"] == case].sort_values("n")
        p0 = cases[case][0]
        ax.plot(sub["n"], sub["rwa_ev"], "-", color="tab:blue", label="Extended Vasicek RWA")
        ax.axhline(sub["rwa_irb"].iloc[0], color="tab:orange", linestyle="--", label="Basel III/IV IRB RWA")
        ax.axhline(sub["rwa_floor"].iloc[0], color="tab:green", linestyle=":", label="72.5% output floor RWA")
        ax.set_xscale("log")
        ax.set_xlabel("n (number of obligors)")
        ax.set_ylabel("RWA")
        ax.set_title(f"Rating {case} (PD = {p0:.4f})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Experiment 4: RWA vs. n, Basel III/IV vs. Extended Vasicek vs. output floor")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="No-op; kept for interface symmetry with the other experiment scripts.")
    parser.add_argument("--quick", action="store_true",
                         help="No-op; this experiment is deterministic and has no Monte Carlo step.")
    args = parser.parse_args()

    df_full = pd.concat(
        [run_case(case, p0, risk_weight_sa, N_GRID_EXP4) for case, (p0, risk_weight_sa) in CASES.items()],
        ignore_index=True,
    )
    logger.info("exp4: computed %d rows over %d cases x %d grid points",
                len(df_full), len(CASES), len(N_GRID_EXP4))

    df_table = df_full[df_full["n"].isin(N_TABLE_EXP4)].reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    full_csv_path = RESULTS_DIR / "exp4_rwa_single_rating.csv"
    df_full.to_csv(full_csv_path, index=False)
    logger.info("Saved %s (%d rows)", full_csv_path, len(df_full))

    table_csv_path = RESULTS_DIR / "exp4_rwa_table.csv"
    df_table.to_csv(table_csv_path, index=False)
    logger.info("Saved %s (%d rows)", table_csv_path, len(df_table))

    plot_path = RESULTS_DIR / "exp4_rwa_single_rating.png"
    make_plot(df_full, CASES, plot_path)
    logger.info("Saved %s", plot_path)

    summary: dict[str, object] = {}
    for case in CASES:
        sub = df_full[df_full["case"] == case]
        floor_binds_below_n = find_floor_binds_below_n(sub)
        summary[case] = {
            "p0": float(sub["p0"].iloc[0]),
            "rho": float(sub["rho"].iloc[0]),
            "rwa_irb": float(sub["rwa_irb"].iloc[0]),
            "rwa_floor": float(sub["rwa_floor"].iloc[0]),
            "floor_binds_below_n": floor_binds_below_n,
        }
        logger.info(
            "exp4 case %s: rwa_irb=%.4f, rwa_floor=%.4f, floor_binds_below_n=%s",
            case, summary[case]["rwa_irb"], summary[case]["rwa_floor"], floor_binds_below_n,
        )
    summary["table"] = df_table.to_dict(orient="records")

    summary_path = RESULTS_DIR / "exp4_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved %s", summary_path)


if __name__ == "__main__":
    main()
