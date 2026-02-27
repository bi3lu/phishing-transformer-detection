from typing import Any

import mlflow
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model(model_uri: str) -> Pipeline:
    """Load a trained model from MLflow or a local directory/file.

    The original implementation assumed an MLflow URI such as
    ``runs:/<run_id>/model``.  To support batch evaluation of models
    stored on disk (e.g. under ``saved_models/``) we detect whether the
    provided URI points to an existing filesystem path and try to load the
    model accordingly.

    Supported cases:

    * MLflow URI (default behaviour).
    * Local directory containing an MLflow model (i.e. an ``MLmodel``
      descriptor).  This is also handled by :meth:`mlflow.sklearn.load_model`
      so no special action is required.
    * Local ``pipeline.joblib`` created by the baseline script.

    Args:
        model_uri: MLflow model URI or filesystem path identifying the
            model to load.

    Returns:
        A scikit-learn ``Pipeline`` object (or other object implementing
        ``predict_proba``) loaded from the specified location.
    """
    logger.info(f"Loading model from: {model_uri}")
    path = None

    try:
        from pathlib import Path

        path = Path(model_uri)

    except Exception:
        path = None

    if path and path.exists():
        try:
            return mlflow.sklearn.load_model(model_uri)

        except Exception:
            joblib_file = path / "pipeline.joblib"
            if joblib_file.exists():
                import joblib

                return joblib.load(joblib_file)
            raise
    else:
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


def evaluate_model(model: Any, X_test: Any, y_test: Any, threshold: float = 0.5) -> None:
    """Evaluate a trained model on a test set.

    Computes predicted probabilities, applies a classification threshold,
    and reports standard binary classification metrics including
    precision, recall, F1-score, ROC-AUC, confusion matrix, and a
    detailed classification report.  If an MLflow run is active, the
    scores are also logged using ``mlflow.log_metrics`` so that batch
    evaluations show up in the tracking server.

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

    # log to MLflow if a run is active
    if mlflow.active_run():
        mlflow.log_metrics({"f1": f1, "precision": precision, "recall": recall, "auc": auc})


def evaluate_predictions(probs: np.ndarray, y_test: Any, threshold: float = 0.5) -> None:
    """Evaluate using pre-computed positive-class probabilities.

    This helper is used when the model object itself cannot be
    executed (for example when evaluating a Hugging Face checkpoint). It
    mirrors the behaviour of :func:`evaluate_model` by computing metrics
    from the probability array and optionally logging them to MLflow.

    Args:
        probs: Array of floats representing the predicted probability of
            the positive class.
        y_test: True labels.
        threshold: Decision threshold.
    """
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

    from sklearn.metrics import confusion_matrix

    logger.info(confusion_matrix(y_test, preds))

    logger.info("Classification report:")
    logger.info("\n" + classification_report(y_test, preds))

    if mlflow.active_run():
        mlflow.log_metrics({"f1": f1, "precision": precision, "recall": recall, "auc": auc})
