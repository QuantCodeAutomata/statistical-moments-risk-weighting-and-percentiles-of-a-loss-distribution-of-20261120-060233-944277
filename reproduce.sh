#!/usr/bin/env bash
# Runs the test suite and every experiment of the replication of
# M. van der Burgt, "Statistical Moments, Risk Weighting and Percentiles of a
# Loss Distribution of a Finite Homogeneous Credit Portfolio; extending the
# Vasicek Distribution" (November 2026), at the paper's configuration
# (500,000 Monte Carlo runs per grid point, Section 4.1).
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Running test suite ==="
python -m pytest || exit 1

echo "=== Experiment 1: central moments vs simulation ==="
python -m exp.exp1

echo "=== Experiment 2: capital multiple comparison ==="
python -m exp.exp2

echo "=== Experiment 3: 99.9% percentile comparison ==="
python -m exp.exp3

echo "=== Experiment 4: single-rating RWA comparison ==="
python -m exp.exp4

echo "=== Experiment 5: multi-rating RWA with concentration effects ==="
python -m exp.exp5

echo "=== Done: see results/RESULTS.md ==="
