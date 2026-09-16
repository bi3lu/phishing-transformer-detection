#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
uv run --frozen python main.py --only finetune
