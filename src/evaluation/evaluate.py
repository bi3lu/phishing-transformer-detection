import argparse
from typing import Tuple

import mlflow  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.metrics import classification_report  # type: ignore
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # type: ignore

from src.config import LABEL_COL, SPLIT_DATA_DIR, TEXT_COL
from src.utils.logger import get_logger

# Setup logging:
logger = get_logger(__name__)


# Helper functions:
def _load_test_data() -> Tuple[pd.Series, pd.Series]:
    """Load test dataset and return features and labels.

    Reads the ``test.csv`` file from ``SPLIT_DATA_DIR``, extracts the
    configured text column as input features and the label column as
    integer targets.

    Returns:
        A tuple (X, y), where:
            - X: pandas Series containing text inputs (strings).
            - y: pandas Series containing integer labels.

    Raises:
        FileNotFoundError: If the test split file does not exist.
    """
    path = SPLIT_DATA_DIR / "test.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing test split: {path}")

    df = pd.read_csv(path)

    X = df[TEXT_COL].astype(str)
    y = df[LABEL_COL].astype(int)

    return (X, y)


def load_model(model_uri: str) -> Pipeline:
    """Load a trained scikit-learn pipeline from MLflow.

    Args:
        model_uri: MLflow model URI pointing to a logged model artifact
            (e.g., ``runs:/<run_id>/model`` or a registry URI).

    Returns:
        A scikit-learn Pipeline object loaded from MLflow.
    """
    logger.info(f"Loading model from: {model_uri}")

    return mlflow.sklearn.load_model(model_uri)


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities to binary predictions using a threshold.

    Args:
        probs: Array of predicted probabilities for the positive class.
        threshold: Decision threshold in the range [0, 1]. Values greater
            than or equal to this threshold are mapped to class 1.

    Returns:
        A NumPy array of integer predictions (0 or 1).
    """
    return (probs >= threshold).astype(int)


# Main:
def main(model_uri: str, threshold: float) -> None:
    """Evaluate a trained model on the test dataset.

    The function loads the test data and a trained model from MLflow,
    computes predicted probabilities, applies a classification threshold,
    and reports standard binary classification metrics including
    precision, recall, F1-score, ROC-AUC, confusion matrix, and a
    detailed classification report.

    Args:
        model_uri: MLflow model URI identifying the model to evaluate.
        threshold: Classification threshold used to convert probabilities
            into binary predictions.
    """
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
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--model_uri",
        type=str,
        required=True,
        help="MLflow model URI, e.g. runs:/<run_id>/model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5).",
    )

    args = parser.parse_args()

    main(model_uri=args.model_uri, threshold=args.threshold)
