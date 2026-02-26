import mlflow  # type: ignore
import numpy as np  # type: ignore
from sklearn.metrics import classification_report  # type: ignore
from sklearn.metrics import (  # type: ignore
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # type: ignore

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model(model_uri: str) -> Pipeline:
    """Load a trained scikit-learn pipeline from MLflow.

    Args:
        model_uri: MLflow model URI pointing to a logged model artifact.

    Returns:
        A scikit-learn Pipeline object loaded from MLflow.
    """
    logger.info(f"Loading model from: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities to binary predictions using a threshold.

    Args:
        probs: Array of predicted probabilities for the positive class.
        threshold: Decision threshold in the range [0, 1].

    Returns:
        A NumPy array of integer predictions (0 or 1).
    """
    return (probs >= threshold).astype(int)


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> None:
    """Evaluate a trained model on a test set.

    Computes predicted probabilities, applies a classification threshold,
    and reports standard binary classification metrics including
    precision, recall, F1-score, ROC-AUC, confusion matrix, and a
    detailed classification report.

    Args:
        model: Trained model (must implement predict_proba).
        X_test: Test features.
        y_test: True labels for the test set.
        threshold: Classification threshold for binary prediction.
    """
    logger.info("Evaluating model...")

    probs = model.predict_proba(X_test)[:, 1]
    preds = apply_threshold(probs, threshold)

    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    logger.info("Test results:")
    logger.info(f"F1: \t{f1:.4f}")
    logger.info(f"Precision: \t{precision:.4f}")
    logger.info(f"Recall: \t{recall:.4f}")
    logger.info(f"ROC-AUC: \t{auc:.4f}")

    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_test, preds))

    logger.info("Classification report:")
    logger.info("\n" + classification_report(y_test, preds))
