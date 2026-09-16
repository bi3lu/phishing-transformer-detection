#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
runner=()
if command -v caffeinate >/dev/null 2>&1; then runner=(caffeinate -i); fi
"${runner[@]}" uv run --frozen python main.py --only preprocess split
for seed in 42 43 44; do
    PHISHING_SEED="$seed" "${runner[@]}" uv run --frozen python main.py --skip preprocess split
done
uv run --frozen python -m src.evaluation.report --across-seeds
