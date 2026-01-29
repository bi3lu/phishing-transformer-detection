import argparse
import logging
from pathlib import Path
from typing import Tuple

import colorlog  # type: ignore
import mlflow  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.metrics import (  # type: ignore
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # type: ignore

# Paths:
BASE_DATA_DIR = Path(__file__).resolve().parents[2]
SPLIT_DATA_DIR = BASE_DATA_DIR / "data" / "split"

# Setup logging:
logger = logging.getLogger(__name__)

if logger.hasHandlers():
    logger.handlers.clear()

logger.setLevel(logging.DEBUG)

log_format = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s%(reset)s",
    datefmt="%H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)

logger.addHandler(console_handler)

# Constants:
LABEL_COL = "Is_Phishing"
TEXT_COL = "Text"


# Helper functions:
def _load_test_data() -> Tuple[pd.Series, pd.Series]:  # TODO: Add docstring
    path = SPLIT_DATA_DIR / "test.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing test split: {path}")

    df = pd.read_csv(path)

    X = df[TEXT_COL].astype(str)
    y = df[LABEL_COL].astype(int)

    return (X, y)


def load_model(model_uri: str) -> Pipeline:  # TODO: Add docstring
    logger.info(f"Loading model from: {model_uri}")

    return mlflow.sklearn.load_model(model_uri)


def apply_threshold(
    probs: np.ndarray, threshold: float
) -> np.ndarray:  # TODO: Add docstring
    return (probs >= threshold).astype(int)


# Main:
def main(model_uri: str, threshold: float) -> None:
    logger.info("Starting evaluation")

    X_test, y_test = _load_test_data()
    model = load_model(model_uri)

    probs = model.predict_proba(X_test)[:, 1]
    preds = apply_threshold(probs, threshold)

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    logger.info("Evaluation results:")
    logger.info(f"Threshold: \t{threshold}")
    logger.info(f"Precision: \t{precision:.4f}")
    logger.info(f"Recall: \t{recall:.4f}")
    logger.info(f"F1-score: \t{f1:.4f}")
    logger.info(f"ROC-AUC: \t{auc:.4f}")

    logger.info("Confusion Matrix:")
    logger.info(f"\n{confusion_matrix(y_test, preds)}")

    logger.info("Classification Report:")
    logger.info("\n" + classification_report(y_test, preds, digits=4))


# Entry point:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-uri",
        type=str,
        required=True,
        help="MLflow model URI, e.g. runs:/<run_id>/model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for positive class",
    )

    args = parser.parse_args()

    main(args.model_uri, args.threshold)
