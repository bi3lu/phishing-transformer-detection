#!/bin/bash
# run_experiments.sh
# Runs fine-tuning for all defined experiments in params.yaml

# List of experiments
EXPERIMENTS=("herbert-base" "polish-roberta-v2" "xlm-roberta-base" "distilbert-multilingual")

echo "Starting experiments: ${EXPERIMENTS[*]}"

for exp in "${EXPERIMENTS[@]}"; do
    echo "Running experiment: $exp"
    
    PYTHONPATH=. python src/models/fine_tune.py --experiment_name "$exp"
    
    if [ $? -ne 0 ]; then
        echo "Error running experiment: $exp"
        exit 1
    fi
    
    echo "Experiment $exp completed successfully."
done

echo "All experiments completed."
