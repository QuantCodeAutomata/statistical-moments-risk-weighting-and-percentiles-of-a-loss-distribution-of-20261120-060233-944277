# Repository guide for coding agents and contributors

This repository replicates M. van der Burgt, "Statistical Moments, Risk
Weighting and Percentiles of a Loss Distribution of a Finite Homogeneous
Credit Portfolio; extending the Vasicek Distribution" (November 2026). It
reproduces the paper's finite-portfolio central-moment formulas (Theorem 2),
its capital-multiple and percentile approximations (Cornish-Fisher expansion
and 1/n-weighting, Section 3.2), and its Extended Vasicek risk-weight formula
(Section 5), all against Monte Carlo simulation of the single-factor
homogeneous portfolio model the paper itself defines (Section 4.1). No
market data are used anywhere in this repository.

## Layout

- `src/` — flat library modules, no subdirectories:
  - `config.py` — every experiment parameter as a module-level constant,
    grouped by experiment, each with its paper section/equation or "not
    specified in the paper".
  - `vasicek.py` — Vasicek-limit building blocks: the BCBS correlation
    formula (Eq. 22), the default threshold, the Vasicek-limit moments
    (Theorem 1, Eq. 5), the Vasicek variance (Eq. 7) and percentile (Eq. 8).
  - `moments.py` — exact finite-portfolio central moments (Theorem 2,
    Eq. 10-12, Appendix B), using `scipy.special.stirling2`.
  - `percentiles.py` — capital multiples and the Extended Vasicek percentile
    (Eq. 13, 15, 19-21), using `scipy.stats.mstats.hdquantiles` for the
    Harrell-Davis estimator.
  - `cornish_fisher.py` — the Cornish-Fisher capital multiple (Eq. 15-18),
    with the central moments of the Cornish-Fisher-expanded variable derived
    symbolically with `sympy` at import time (the paper does not reproduce
    these closed forms, citing J.D. Lamb, 2016, for them).
  - `rwa.py` — Basel III/IV IRB, Extended Vasicek and simulated RWA, including
    the multi-rating aggregation of Eq. 23-24 (Section 5).
- `exp/` — one script per experiment (`exp1.py` .. `exp5.py`), run as
  `python -m exp.expN`; each writes its tables, figures and a JSON summary
  directly into `results/`.
- `data/generate_synthetic.py` — the single-factor Monte Carlo sampler
  (Section 4.1) shared by every experiment; conditional on the systematic
  factor, the default count is drawn directly from its exact conditional
  binomial distribution, which reproduces the paper's per-obligor simulation
  exactly while remaining tractable for n up to 2000 and 500,000 runs.
- `tests/` — pytest unit tests of `src/` and `data/` on synthetic fixtures
  and known analytic values (binomial moments, bivariate normal properties).
- `results/` — every table, figure and the mandatory `results/RESULTS.md`,
  written directly here (no per-experiment subdirectories). Rerunning an
  experiment overwrites its files; `results/RESULTS.md` is not regenerated
  automatically and must be checked against the new files and updated by
  hand after a rerun.

## Commands

```bash
pip install -r requirements.txt
python -m pytest                     # all tests are synthetic/analytic, none need a market_data marker
python -m exp.exp1                   # ... through exp5, paper configuration (500,000 Monte Carlo runs)
python -m exp.exp1 --quick           # fast smoke test with a reduced run count
bash reproduce.sh                    # tests, then all five experiments at the paper configuration
```

## Conventions

- Every experiment parameter lives only in `src/config.py`, as module-level
  `UPPER_CASE` constants with a comment giving its paper source; no classes
  or parameter dictionaries.
- Data preparation code lives only in `data/generate_synthetic.py`; no
  experiment script generates its own samples.
- All pytest files live in `tests/`; they exercise the paper's qualitative
  properties (signs, convergence to the Vasicek limit, absence of
  look-ahead in the moment decomposition, admissible ranges) rather than
  reproducing the paper's exact reported numbers, which are instead checked
  in `results/RESULTS.md` against each experiment's saved output.
- No API keys are used anywhere in this repository; there is no `.env` file
  and no `.env.example`, since no experiment loads market data.
- After rerunning an experiment with `python -m exp.expN`, its files in
  `results/` are overwritten. The numbers, tables and verdicts quoted in
  `results/RESULTS.md` are not regenerated automatically: check them against
  the new files and update `results/RESULTS.md` by hand.
