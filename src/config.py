"""Parameters for the replication of van der Burgt, "Statistical Moments, Risk
Weighting and Percentiles of a Loss Distribution of a Finite Homogeneous
Credit Portfolio; extending the Vasicek Distribution" (November 2026).

All experiment parameters are collected here as module-level constants, grouped
by the experiment that uses them. Each constant carries the paper section or
equation it comes from, or states that the paper does not specify it.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Common simulation set-up (Section 4.1)
# ---------------------------------------------------------------------------

# Number of Monte Carlo runs per (P0, n) configuration (Section 4.1: "based on
# 500,000 simulations"). The single-factor conditional-binomial sampler (see
# data/generate_synthetic.py) draws the conditional binomial count directly,
# which keeps the full paper count tractable within the available computation
# time (the complete grid of Experiment 1 runs in well under a minute), so it
# is used as the default for every experiment below.
N_SIMS = 500_000
# Reduced Monte Carlo run count used only by the --quick flag of the
# experiment scripts for fast development smoke-testing; the reported results
# always use N_SIMS (the paper value), never this one.
N_SIMS_RUN = 50_000

# Confidence level alpha used throughout Sections 3.2, 4 and 5.
ALPHA = 0.999

# Seed for the numpy random generator; the paper (RStudio simulations) does
# not specify a seed, so this value is fixed only for reproducibility of this
# implementation.
RANDOM_SEED = 20261120

# The common grid of obligor counts named in Section 4.1 ("portfolios
# consisting of 2 to 2000 obligors") and illustrated by the explicit n-values
# listed in experiment descriptions of Figures 1-3, 5 and 6.
N_GRID_FULL = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800,
    900, 1000, 2000,
]

# ---------------------------------------------------------------------------
# Experiment 1: verification of central moments (Section 3.1, Theorem 2;
# Section 4.1-4.2, Figures 1-3)
# ---------------------------------------------------------------------------

# Unconditional default probabilities used in the simulation set-up (Section
# 4.1: "P0, ranging from 0.0005 to 0.02").
P0_GRID_EXP1 = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]

N_GRID_EXP1 = N_GRID_FULL

# ---------------------------------------------------------------------------
# Experiment 2: capital multiple comparison (Section 3.2, Figures 4-5, Table 4)
# ---------------------------------------------------------------------------

# P0 grid spanning the panels of Figure 5 and Table 4.
P0_GRID_EXP2 = [0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

N_GRID_EXP2 = N_GRID_FULL

# ---------------------------------------------------------------------------
# Experiment 3: 99.9% percentile comparison (Section 3.2.2, Figure 6)
# ---------------------------------------------------------------------------

P0_GRID_EXP3 = [0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

# Figure 6 is drawn for n from 2 to 200 (experiment description, data_requirements).
N_GRID_EXP3 = [n for n in N_GRID_FULL if n <= 200]

# ---------------------------------------------------------------------------
# Experiment 4: single rating-grade RWA comparison (Section 5, Figure 7)
# ---------------------------------------------------------------------------

# PD of an S&P-A rated obligor (Section 5: "PD of 0.04%"), sourced in the
# paper from the latest observed S&P default rates (N. Serino, 2025).
RATING_A_PD = 0.0004
# Standardised-approach risk weight for rating A (Section 5): 50%.
RATING_A_RISK_WEIGHT_SA = 0.50

# PD of an S&P-BB rated obligor (Section 5: "PD of 0.64%").
RATING_BB_PD = 0.0064
# Standardised-approach risk weight used in the 72.5%-floor computation for
# rating BB. Section 5 states the floor as "72.5% x 100% x 100 = 72.50",
# which requires a 100% risk weight; the running text in the same paragraph
# instead says "50%". This implementation follows the floor value the paper
# actually reports (100%); the inconsistency is recorded in
# results/RESULTS.md ("Deviations from the paper").
RATING_BB_RISK_WEIGHT_SA = 1.00

# Loss given default and exposure at default used for both ratings
# (Section 5: "LGD of 0.4 and an EAD of 100").
LGD_EXP4 = 0.4
EAD_EXP4 = 100.0

# Output floor applied to the standardised-approach RWA (Section 5: "72.5%").
OUTPUT_FLOOR = 0.725

# Obligor counts for the Table-style comparison in Figure 7's caption table.
N_TABLE_EXP4 = [10, 100, 1000]
# Continuous grid used to draw Figure 7 (log-scale in n); not individually
# listed in the paper, which plots a smooth curve.
N_GRID_EXP4 = sorted(set(N_GRID_FULL) | {1, 2})

# ---------------------------------------------------------------------------
# Experiment 5: multi-rating RWA comparison with concentration effects
# (Section 5, Table 5, Table 6)
# ---------------------------------------------------------------------------

RATINGS = ["A", "B", "C", "D"]

# PD per rating grade (Table 5).
PD_BY_RATING = {"A": 0.001, "B": 0.002, "C": 0.015, "D": 0.020}

# EAD (EUR million) and LGD, identical for every rating grade (Table 5).
EAD_EXP5 = 100.0
LGD_EXP5 = 0.4

# Asset correlation matrix between rating grades (Table 5), used only to
# convert into default correlations via Eq. (24) for the analytical
# Extended-Vasicek aggregation; it is not the within-rating correlation used
# to drive the Monte Carlo simulation (that comes from Eq. 22, see Section
# 5 and the experiment's implementation_notes).
ASSET_CORRELATION_TABLE5 = {
    ("A", "A"): 1.00, ("A", "B"): 0.25, ("A", "C"): 0.15, ("A", "D"): 0.10,
    ("B", "A"): 0.25, ("B", "B"): 1.00, ("B", "C"): 0.17, ("B", "D"): 0.19,
    ("C", "A"): 0.15, ("C", "B"): 0.17, ("C", "C"): 1.00, ("C", "D"): 0.22,
    ("D", "A"): 0.10, ("D", "B"): 0.19, ("D", "C"): 0.22, ("D", "D"): 1.00,
}

# Portfolio compositions Pi_1 .. Pi_6: obligor counts per rating grade
# (Table 6, first four rows).
PORTFOLIOS_TABLE6 = {
    "Pi_1": {"A": 10, "B": 10, "C": 10, "D": 10},
    "Pi_2": {"A": 6, "B": 6, "C": 21, "D": 7},
    "Pi_3": {"A": 5, "B": 5, "C": 5, "D": 25},
    "Pi_4": {"A": 100, "B": 8, "C": 8, "D": 8},
    "Pi_5": {"A": 200, "B": 10, "C": 8, "D": 4},
    "Pi_6": {"A": 260, "B": 10, "C": 26, "D": 4},
}

# Reported RWA values from Table 6, used only to populate the "paper value"
# column of results/RESULTS.md; not used anywhere in the computation.
TABLE6_REPORTED_RWA = {
    "Pi_1": {"BIII_IV": 2052, "sim": 2811, "EV": 2779},
    "Pi_2": {"BIII_IV": 2467, "sim": 3266, "EV": 3268},
    "Pi_3": {"BIII_IV": 2728, "sim": 3703, "EV": 3649},
    "Pi_4": {"BIII_IV": 3168, "sim": 3632, "EV": 3464},
    "Pi_5": {"BIII_IV": 4541, "sim": 4792, "EV": 4745},
    "Pi_6": {"BIII_IV": 6918, "sim": 6638, "EV": 6334},
}

# Target Herfindahl-Hirschman Index values of the concentration study in
# Figure 8 (Section 5), all at a fixed n_Total=40. The paper gives these HHI
# values but not the underlying obligor compositions, so the composition
# producing each HHI is constructed here (see exp/exp5.py): starting from the
# equal split (A=B=C=D=10, HHI=0.25, as in Pi_1) and concentrating the
# remaining obligors into rating D, the rating with the highest PD among the
# four, which reproduces the direction of concentration seen from Pi_1 to
# Pi_3 in Table 6.
HHI_TARGETS_EXP5 = [0.25, 0.28, 0.30, 0.35, 0.40, 0.43, 0.52, 0.59]
N_TOTAL_HHI_EXP5 = 40
