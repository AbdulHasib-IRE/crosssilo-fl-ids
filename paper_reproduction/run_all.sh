#!/usr/bin/env bash
# Recomputes every derived statistic and figure from the result files in data/.
# Requires python>=3.10 with numpy, pandas, scipy, matplotlib. About 1 min on a laptop.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"   # override with PYTHON=python if needed
mkdir -p outputs/analysis outputs/figures
"$PY" scripts/facts_xs.py data/export data data outputs/analysis/facts_xs.json > outputs/analysis/standard_metrics.txt
"$PY" scripts/make_xs_figures.py data/export data outputs/analysis/facts_xs.json outputs/figures
echo "done: figures in outputs/figures, statistics in outputs/analysis"
