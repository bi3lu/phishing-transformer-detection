"""Main pipeline orchestrator for phishing detection.

Runs the full pipeline end-to-end: data preprocessing, splitting,
baseline training, transformer fine-tuning, evaluation, threshold
analysis, and ensemble inference.

Usage:
    python main.py                                                  # Run everything
    python main.py --skip-preprocess                                # Skip data preprocessing
    python main.py --only baseline                                  # Run only baseline
    python main.py --experiments herbert-base polish-roberta-v2
"""

import argparse
import sys
import time
from typing import List

import yaml

from src.config import BASE_DIR, SPLIT_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("main")

STEPS = [
    "preprocess",
    "split",
    "baseline",
    "finetune",
    "evaluate",
    "threshold",
    "ensemble",
    "kfold",
    "analysis",
]


def step_preprocess() -> None:
    """Step 1: Preprocess raw data into structured CSV."""
    logger.info("=" * 60)
    logger.info("STEP 1/9: Preprocessing raw data...")
    logger.info("=" * 60)

    from src.data.preprocess_data import main as preprocess_main

    preprocess_main()


def step_split() -> None:
    """Step 2: Split data into train/val/test with feature extraction."""
    logger.info("=" * 60)
    logger.info("STEP 2/9: Splitting data & extracting features...")
    logger.info("=" * 60)

    from src.data.split_data import main as split_main

    split_main()


def step_baseline() -> None:
    """Step 3: Train baseline TF-IDF + Logistic Regression."""
    logger.info("=" * 60)
    logger.info("STEP 3/9: Training baseline model...")
    logger.info("=" * 60)

    from src.models.baseline import main as baseline_main

    baseline_main()


def step_finetune(experiments: List[str]) -> None:
    """Step 4: Fine-tune transformer models."""
    logger.info("=" * 60)
    logger.info(f"STEP 4/9: Fine-tuning transformers: {experiments}")
    logger.info("=" * 60)

    import gc

    import torch

    from src.models.fine_tune import main as finetune_main

    for exp_name in experiments:
        logger.info(f"--- Starting experiment: {exp_name} ---")
        start = time.time()

        try:
            finetune_main(experiment_name=exp_name)
            elapsed = time.time() - start
            logger.info(f"--- Experiment {exp_name} done ({elapsed:.0f}s) ---")

        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")

        finally:
            # Free GPU/MPS memory between experiments:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()


def step_evaluate() -> None:
    """Step 5: Evaluate all saved models on test set."""
    logger.info("=" * 60)
    logger.info("STEP 5/9: Evaluating all models...")
    logger.info("=" * 60)

    from src.evaluation.evaluate import main as evaluate_main

    evaluate_main(models_dir=str(BASE_DIR / "saved_models"), threshold=0.5)


def step_threshold() -> None:
    """Step 6: Threshold analysis across all models."""
    logger.info("=" * 60)
    logger.info("STEP 6/9: Running threshold analysis...")
    logger.info("=" * 60)

    from src.evaluation.threshold_analysis import main as threshold_main

    threshold_main()


def step_ensemble(threshold: float) -> None:
    """Step 7: Ensemble inference."""
    logger.info("=" * 60)
    logger.info("STEP 7/9: Running ensemble inference...")
    logger.info("=" * 60)

    from src.evaluation.ensemble import main as ensemble_main

    ensemble_main(threshold=threshold)


def step_kfold(experiments: List[str]) -> None:
    """Step 8: K-Fold cross-validation."""
    logger.info("=" * 60)
    logger.info(f"STEP 8/9: Running K-Fold CV: {experiments}")
    logger.info("=" * 60)

    import gc

    import torch

    from src.models.kfold_cv import run_kfold

    for exp_name in experiments:
        logger.info(f"--- K-Fold CV for {exp_name} ---")
        start = time.time()

        try:
            run_kfold(experiment_name=exp_name, n_splits=5)
            elapsed = time.time() - start
            logger.info(f"--- K-Fold {exp_name} done ({elapsed:.0f}s) ---")
        except Exception as e:
            logger.error(f"K-Fold {exp_name} failed: {e}")
        finally:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()


def step_analysis() -> None:
    """Step 9: Advanced analysis (error, distributions, McNemar, ablation)."""
    logger.info("=" * 60)
    logger.info("STEP 9/9: Running advanced analysis...")
    logger.info("=" * 60)

    from src.evaluation.analysis import main as analysis_main

    analysis_main()


def get_experiment_names() -> List[str]:
    """Read all experiment names from params.yaml."""
    params_path = BASE_DIR / "params.yaml"

    with open(params_path, "r") as f:
        config = yaml.safe_load(f)

    return [exp["name"] for exp in config["experiments"]]


# Main:
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full phishing detection pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  PYTHONPATH=. python main.py                         # Full pipeline
  PYTHONPATH=. python main.py --skip preprocess split  # Skip data prep
  PYTHONPATH=. python main.py --only finetune evaluate # Only train & eval
  PYTHONPATH=. python main.py --experiments herbert-base
  PYTHONPATH=. python main.py --threshold 0.4
        """,
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=STEPS,
        default=[],
        help="Steps to skip",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=STEPS,
        default=[],
        help="Run only these steps (overrides --skip)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[],
        help="Transformer experiments to fine-tune (default: all from params.yaml)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Classification threshold for evaluation and ensemble (default: 0.35)",
    )
    args = parser.parse_args()

    # Determine which steps to run:
    if args.only:
        steps_to_run = args.only

    else:
        steps_to_run = [s for s in STEPS if s not in args.skip]

    experiments = args.experiments if args.experiments else get_experiment_names()

    logger.info("=" * 60)
    logger.info("PHISHING DETECTION PIPELINE")
    logger.info(f"Steps:       {steps_to_run}")
    logger.info(f"Experiments: {experiments}")
    logger.info(f"Threshold:   {args.threshold}")
    logger.info("=" * 60)

    pipeline_start = time.time()

    if "preprocess" in steps_to_run:
        step_preprocess()

    if "split" in steps_to_run:
        step_split()

    # Verify splits exist before training:
    if any(s in steps_to_run for s in ["baseline", "finetune", "evaluate", "threshold", "ensemble"]):
        if not (SPLIT_DATA_DIR / "train.csv").exists():
            logger.error("Split files not found. Run preprocess and split steps first.")
            sys.exit(1)

    if "baseline" in steps_to_run:
        step_baseline()

    if "finetune" in steps_to_run:
        step_finetune(experiments)

    if "evaluate" in steps_to_run:
        step_evaluate()

    if "threshold" in steps_to_run:
        step_threshold()

    if "ensemble" in steps_to_run:
        step_ensemble(args.threshold)

    if "kfold" in steps_to_run:
        step_kfold(experiments)

    if "analysis" in steps_to_run:
        step_analysis()

    elapsed = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE ({elapsed:.0f}s)")
    logger.info("=" * 60)


# Entry point
if __name__ == "__main__":
    main()
