"""Baseline phishing detection using TF-IDF and Logistic Regression.

Trains and evaluates a classical machine learning pipeline as a baseline
for comparison with transformer-based models."""

import os
from typing import Any, Dict

import joblib
import mlflow
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import BASE_DIR, RANDOM_STATE, SAVED_MODELS_DIR
from src.data.load_data import load_split, prepare_xy
from src.models.provenance import write_training_manifest
from src.utils.logger import get_logger

# Setup logging:
logger = get_logger(__name__)


def build_baseline_pipeline(config: Dict[str, Any]) -> Pipeline:
    """Create the canonical unfitted baseline used by training and CV."""
    tfidf_params = config["baseline"]["tfidf"]
    lr_params = config["baseline"]["logistic_regression"]
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(tfidf_params["ngram_range"]),
                    max_features=tfidf_params["max_features"],
                    min_df=tfidf_params["min_df"],
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=lr_params["max_iter"],
                    class_weight=lr_params["class_weight"],
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# Logging metrics:
def log_metrics(y_true: Any, y_pred: Any, y_probs: Any, prefix: str = "val") -> None:
    """Compute and log classification metrics.

    Calculates F1, precision, recall, and ROC-AUC scores, logs them using
    the logger and MLflow if a run is active.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_probs: Predicted probabilities for the positive class.
        prefix: Metric prefix for logging (e.g., 'val', 'test').
    """
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)

    display_name = "Validation" if prefix == "val" else prefix.capitalize()

    logger.info(f"{display_name} results:")
    logger.info(f"F1: \t{f1:.4f}")
    logger.info(f"Precision: \t{precision:.4f}")
    logger.info(f"Recall: \t{recall:.4f}")
    logger.info(f"ROC-AUC: \t{auc:.4f}")

    mlflow.log_metrics(
        {
            f"{prefix}_f1": f1,
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
            f"{prefix}_auc": auc,
        }
    )


# Main:
def main() -> None:
    """Train once on train; validation remains exclusively for downstream decisions."""
    from src.models.provenance import artefact_matches_current_split, current_split_manifest
    from src.utils.artifacts import bind_run, identity, stage_status

    bind_run()
    config = yaml.safe_load((BASE_DIR / "params.yaml").read_text())
    destination = SAVED_MODELS_DIR / "baseline"

    if artefact_matches_current_split(destination):
        return

    training = load_split("train")
    x_train, y_train = prepare_xy(training)

    with stage_status(destination, identity({"split": current_split_manifest(), "config": config["baseline"]})):
        pipeline = build_baseline_pipeline(config)
        pipeline.fit(x_train, y_train)
        joblib.dump(pipeline, destination / "pipeline.joblib")
        write_training_manifest(destination, "baseline", config["baseline"])


if __name__ == "__main__":
    main()
