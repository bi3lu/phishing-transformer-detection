#!/usr/bin/env bash
# run_experiments.sh
# Runs fine-tuning for all defined experiments in params.yaml

# ensure we exit on error
set -e

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# List of experiments
EXPERIMENTS=("herbert-base" "polish-roberta-v2" "xlm-roberta-base" "distilbert-multilingual")

echo "Starting experiments: ${EXPERIMENTS[*]}"

for exp in "${EXPERIMENTS[@]}"; do
    echo "Running experiment: $exp"

    # use python3 as a fallback if python is not in PATH
    PYTHONPATH=. $(command -v python || command -v python3) src/models/fine_tune.py --experiment_name "$exp"

    echo "Experiment $exp completed successfully."
done

echo "All experiments completed."
