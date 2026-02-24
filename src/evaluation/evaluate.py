import argparse
from typing import Tuple

import pandas as pd  # type: ignore

from src.config import LABEL_COL, SPLIT_DATA_DIR, TEXT_COL
from src.models.utils import evaluate_model, load_model
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

    evaluate_model(model, X_test, y_test, threshold)


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
