"""Experiment 5: RWA for multi-rating portfolios with concentration effects
(Section 5, Table 5, Table 6; Figure 8), comparing simulation, Basel III/IV
IRB and the analytical Extended Vasicek aggregation.

Run: python -m exp.exp5 [--full] [--quick]
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

from data.generate_synthetic import simulate_multirating_portfolio
from src.config import (
    ALPHA,
    ASSET_CORRELATION_TABLE5,
    EAD_EXP5,
    HHI_TARGETS_EXP5,
    LGD_EXP5,
    N_SIMS,
    N_SIMS_RUN,
    N_TOTAL_HHI_EXP5,
    PD_BY_RATING,
    PORTFOLIOS_TABLE6,
    RANDOM_SEED,
    RATINGS,
    TABLE6_REPORTED_RWA,
)
from src.rwa import (
    herfindahl_hirschman_index,
    rwa_basel_irb_multirating,
    rwa_extended_vasicek_multirating,
    rwa_simulated,
)
from src.vasicek import rho_from_pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def rho_by_rating() -> dict[str, float]:
    """Within-rating asset correlation rho_I from the BCBS formula (Eq. 22),
    used to drive the single-factor Monte Carlo simulation and the per-rating
    Extended Vasicek RWA (Section 5, implementation_notes)."""
    return {r: rho_from_pd(PD_BY_RATING[r]) for r in RATINGS}


def evaluate_portfolio(
    composition: dict[str, int], n_sims: int, rng: np.random.Generator
) -> dict:
    """Compute RWA_sim, RWA_BIII/IV and RWA_EV for one portfolio composition
    (Section 5, steps 4-7)."""
    rho_I = rho_by_rating()
    n_total = sum(composition.values())

    samples = simulate_multirating_portfolio(composition, PD_BY_RATING, rho_I, n_sims, rng)
    rwa_sim = rwa_simulated(samples, ALPHA, n_total, LGD_EXP5, EAD_EXP5)

    rwa_biii_iv = rwa_basel_irb_multirating(composition, PD_BY_RATING, rho_I, ALPHA, LGD_EXP5, EAD_EXP5)

    rwa_ev = rwa_extended_vasicek_multirating(
        composition, PD_BY_RATING, rho_I, ASSET_CORRELATION_TABLE5, ALPHA, LGD_EXP5, EAD_EXP5
    )

    hhi = herfindahl_hirschman_index(composition)
    return {
        "n_total": n_total,
        "hhi": hhi,
        "rwa_sim": rwa_sim,
        "rwa_biii_iv": rwa_biii_iv,
        "rwa_ev": rwa_ev,
        **{f"n_{r}": composition.get(r, 0) for r in RATINGS},
    }


def run_table6_comparison(n_sims: int) -> pd.DataFrame:
    """Reproduce the portfolio comparison of Table 6 for Pi_1 .. Pi_6."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    t0 = time.time()
    for name, composition in PORTFOLIOS_TABLE6.items():
        row = evaluate_portfolio(composition, n_sims, rng)
        reported = TABLE6_REPORTED_RWA[name]
        row.update({
            "portfolio": name,
            "rwa_biii_iv_paper": reported["BIII_IV"],
            "rwa_sim_paper": reported["sim"],
            "rwa_ev_paper": reported["EV"],
        })
        rows.append(row)
    logger.info("exp5 Table 6: %d portfolios, n_sims=%d, %.1fs", len(rows), n_sims, time.time() - t0)
    cols = ["portfolio", "n_total", "hhi"] + [f"n_{r}" for r in RATINGS] + [
        "rwa_sim", "rwa_biii_iv", "rwa_ev",
        "rwa_sim_paper", "rwa_biii_iv_paper", "rwa_ev_paper",
    ]
    return pd.DataFrame(rows)[cols]


def build_concentration_composition(n_total: int, n_d: int) -> dict[str, int]:
    """Portfolio with n_d obligors in rating D (the highest-PD rating) and the
    remaining n_total - n_d obligors split as evenly as possible across A, B
    and C. Used to sweep the Herfindahl-Hirschman Index for the concentration
    study of Figure 8 (Section 5), since the paper gives the target HHI
    values but not the underlying compositions (see src/config.py,
    HHI_TARGETS_EXP5)."""
    n_other_total = n_total - n_d
    base, remainder = divmod(n_other_total, 3)
    others = ["A", "B", "C"]
    counts = {r: base for r in others}
    for r in others[:remainder]:
        counts[r] += 1
    counts["D"] = n_d
    return counts


def run_hhi_sweep(n_sims: int, hhi_targets: list[float], n_total: int) -> pd.DataFrame:
    """Scan compositions of n_total obligors concentrating into rating D and
    pick, for each target HHI, the composition whose achieved HHI is
    closest, then evaluate all three RWA methods on it (Figure 8)."""
    candidates = []
    for n_d in range(n_total // 4, n_total - 2):
        composition = build_concentration_composition(n_total, n_d)
        hhi = herfindahl_hirschman_index(composition)
        candidates.append((hhi, composition))

    rng = np.random.default_rng(RANDOM_SEED + 1)
    rows = []
    t0 = time.time()
    used_hhi = set()
    for target in hhi_targets:
        hhi, composition = min(
            (c for c in candidates if round(c[0], 6) not in used_hhi),
            key=lambda c: abs(c[0] - target),
        )
        used_hhi.add(round(hhi, 6))
        row = evaluate_portfolio(composition, n_sims, rng)
        row["hhi_target"] = target
        rows.append(row)
    logger.info("exp5 HHI sweep: %d points, n_sims=%d, %.1fs", len(rows), n_sims, time.time() - t0)
    cols = ["hhi_target"] + ["n_total", "hhi"] + [f"n_{r}" for r in RATINGS] + [
        "rwa_sim", "rwa_biii_iv", "rwa_ev",
    ]
    return pd.DataFrame(rows)[cols].sort_values("hhi_target").reset_index(drop=True)


def make_table6_plot(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: RWA_sim, RWA_BIII/IV, RWA_EV (computed vs. the
    paper's reported values) per portfolio (Table 6)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.12
    methods = [
        ("rwa_biii_iv", "rwa_biii_iv_paper", "BIII/IV"),
        ("rwa_sim", "rwa_sim_paper", "Simulation"),
        ("rwa_ev", "rwa_ev_paper", "Extended Vasicek"),
    ]
    for i, (col, paper_col, label) in enumerate(methods):
        ax.bar(x + (i - 1) * 2 * width, df[col], width, label=f"{label} (this run)")
        ax.bar(x + (i - 1) * 2 * width + width, df[paper_col], width, alpha=0.5,
               label=f"{label} (paper, Table 6)", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(df["portfolio"])
    ax.set_ylabel("RWA")
    ax.set_title("Multi-rating RWA: simulation, Basel III/IV, Extended Vasicek vs. paper (Table 6)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_hhi_plot(df: pd.DataFrame, out_path: Path) -> None:
    """RWA as a function of the Herfindahl-Hirschman Index, for a fixed
    n_Total=40 (Figure 8)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["hhi"], df["rwa_sim"], "o-", label="RWA_sim")
    ax.plot(df["hhi"], df["rwa_biii_iv"], "s-", label="RWA_BIII/IV")
    ax.plot(df["hhi"], df["rwa_ev"], "^-", label="RWA_EV")
    ax.set_xlabel("Herfindahl-Hirschman Index")
    ax.set_ylabel("RWA")
    ax.set_title(f"RWA vs. concentration (n_Total={df['n_total'].iloc[0]})")
    ax.legend()
    ax.grid(True, alpha=0.3)
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    table6 = run_table6_comparison(n_sims)
    table6.to_csv(RESULTS_DIR / "exp5_table6_comparison.csv", index=False)
    make_table6_plot(table6, RESULTS_DIR / "exp5_table6_comparison.png")
    logger.info("Saved exp5_table6_comparison.csv/.png (%d portfolios)", len(table6))

    hhi_sweep = run_hhi_sweep(n_sims, HHI_TARGETS_EXP5, N_TOTAL_HHI_EXP5)
    hhi_sweep.to_csv(RESULTS_DIR / "exp5_hhi_sweep.csv", index=False)
    make_hhi_plot(hhi_sweep, RESULTS_DIR / "exp5_rwa_vs_hhi.png")
    logger.info("Saved exp5_hhi_sweep.csv and exp5_rwa_vs_hhi.png (%d points)", len(hhi_sweep))

    table6["rel_err_sim_vs_paper"] = (table6["rwa_sim"] - table6["rwa_sim_paper"]).abs() / table6["rwa_sim_paper"]
    table6["rel_err_ev_vs_paper"] = (table6["rwa_ev"] - table6["rwa_ev_paper"]).abs() / table6["rwa_ev_paper"]
    table6["rel_err_biii_iv_vs_paper"] = (
        (table6["rwa_biii_iv"] - table6["rwa_biii_iv_paper"]).abs() / table6["rwa_biii_iv_paper"]
    )
    summary = {
        "n_sims": n_sims,
        "table6": table6.to_dict(orient="records"),
        "hhi_sweep_monotone_increasing_ev": bool(np.all(np.diff(hhi_sweep["rwa_ev"]) >= -1e-6)),
        "hhi_targets": HHI_TARGETS_EXP5,
        "n_total_hhi": N_TOTAL_HHI_EXP5,
    }
    pd.Series({"n_sims": n_sims}).to_json(RESULTS_DIR / "exp5_summary.json")
    import json
    with open(RESULTS_DIR / "exp5_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info("exp5 summary written")


if __name__ == "__main__":
    main()
