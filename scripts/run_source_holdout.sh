#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
runner=()
if command -v caffeinate >/dev/null 2>&1; then runner=(caffeinate -i); fi
"${runner[@]}" uv run --frozen python main.py --only source-holdout
