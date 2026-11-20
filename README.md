# Statistical Moments, Risk Weighting and Percentiles of a Loss Distribution of a Finite Homogeneous Credit Portfolio; extending the Vasicek Distribution

Replication of the paper by M. van der Burgt, "Statistical Moments, Risk
Weighting and Percentiles of a Loss Distribution of a Finite Homogeneous
Credit Portfolio; extending the Vasicek Distribution" (November 2026).

The paper extends the Vasicek asymptotic single-factor model to portfolios
with a finite number of obligors `n`. It derives closed-form expressions for
the exact variance, 3rd and 4th central moments of the loss fraction of a
finite homogeneous portfolio (Theorem 2), a weighting scheme and a
Cornish-Fisher expansion for the 99.9% percentile and the associated capital
multiple, and an Extended Vasicek risk-weight formula that converges to the
Basel III/IV IRB formula as `n -> infinity`. All data used here are generated
synthetically by Monte Carlo simulation of the single-factor model that the
paper itself specifies (Section 4.1); no market data are used.

## Repository layout

```
src/            Vasicek-limit and finite-portfolio formulas (flat modules)
exp/            One script per experiment, run as `python -m exp.expN`
data/           Monte Carlo generator of the single-factor model (Section 4.1)
tests/          pytest unit tests for src/ and data/
results/        Tables, figures and RESULTS.md written by the experiments
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys or external data are required: every experiment is driven by
Monte Carlo simulation of the model defined in Section 2 and Section 4.1 of
the paper.

## Running the tests

```bash
python -m pytest
```

All tests use synthetic Monte Carlo samples or analytic fixtures; none call
an external service, so there is no `market_data` marker to skip.

## Running the experiments

Each experiment runs standalone from the repository root and writes its
tables, figures and a JSON summary directly into `results/`:

```bash
python -m exp.exp1   # Verification of central moments (Theorem 2) vs simulation
python -m exp.exp2   # Capital multiple: simulation vs 1/n-weighting vs Cornish-Fisher
python -m exp.exp3   # 99.9% percentile: Extended Vasicek vs Vasicek vs Cornish-Fisher vs simulation
python -m exp.exp4   # Single-rating RWA: Basel III/IV vs Extended Vasicek vs 72.5% floor
python -m exp.exp5   # Multi-rating RWA with concentration effects (Table 5, Table 6)
```

Each script defaults to the paper's configuration of 500,000 Monte Carlo
runs per `(P0, n)` grid point (`N_SIMS` in `src/config.py`, Section 4.1:
"based on 500,000 simulations"). Pass `--quick` to use the smaller
`N_SIMS_RUN` grid instead, for a fast development smoke test; reported
results always come from the default (paper) configuration.

```bash
python -m reproduce.sh   # not a Python module; run it directly, see below
```

or simply:

```bash
bash reproduce.sh
```

which runs the test suite followed by all five experiments at the paper's
configuration.

## Results

See `results/RESULTS.md` for the comparison of every experiment's output
against the values the paper reports, the verdict for each metric, and the
"Deviations from the paper" and "Not reproduced" sections.
