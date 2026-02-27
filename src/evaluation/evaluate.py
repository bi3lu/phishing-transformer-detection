import argparse
from typing import Tuple

import mlflow
import pandas as pd

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


from pathlib import Path


# Main:
def main(model_uri: str = "", models_dir: str = "", threshold: float = 0.5) -> None:
    """Evaluate one or more trained models on the test dataset.

    There are two mutually exclusive modes of operation:

    1. ``model_uri`` is provided (classic behaviour). The specified MLflow
       URI is resolved and the single model evaluated.
    2. ``models_dir`` is provided. All subpaths contained in the directory
       will be treated as individual models and evaluated in turn.  An
       MLflow run will be created for each entry so metrics for each model
       appear separately in the tracking server.

    Either ``model_uri`` or ``models_dir`` must be supplied.

    Args:
        model_uri: MLflow model URI identifying the model to evaluate.
        models_dir: Filesystem path containing multiple models to evaluate.
        threshold: Classification threshold used to convert probabilities
            into binary predictions.
    """
    logger.info("Starting evaluation")

    if model_uri is None and models_dir is None:
        raise ValueError("Either --model_uri or --models_dir must be provided")

    X_test, y_test = _load_test_data()

    if models_dir is not None:
        base = Path(models_dir)

        if not base.exists() or not base.is_dir():
            raise ValueError(f"Provided models_dir is not a directory: {base}")

        for entry in sorted(base.iterdir()):
            if entry.name.startswith("."):
                continue

            run_name = entry.name
            logger.info(f"Evaluating directory entry: {entry}")

            mlflow.set_experiment("phishing_evaluation")

            with mlflow.start_run(run_name=run_name):
                try:
                    model = load_model(str(entry))
                    evaluate_model(model, X_test, y_test, threshold)

                except Exception as e:
                    logger.warning(f"Standard loading failed for {entry}: {e}, trying transformer loader")

                    try:
                        from src.evaluation.threshold_analysis import (
                            load_predictions_transformer,
                        )

                        texts = X_test.tolist()
                        probs = load_predictions_transformer(str(entry), texts)

                        from src.models.utils import evaluate_predictions

                        evaluate_predictions(probs, y_test, threshold)

                    except Exception as e2:
                        logger.error(f"Also failed to evaluate transformer model {entry}: {e2}")
                    continue
    else:
        mlflow.set_experiment("phishing_evaluation")

        with mlflow.start_run():
            model = load_model(model_uri)
            evaluate_model(model, X_test, y_test, threshold)


# Entry point:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model or a directory of models.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model_uri",
        type=str,
        help="MLflow model URI, e.g. runs:/<run_id>/model",
    )
    group.add_argument(
        "--models_dir",
        type=str,
        help="Filesystem directory containing one or more saved models to evaluate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5).",
    )

    args = parser.parse_args()

    main(model_uri=args.model_uri, models_dir=args.models_dir, threshold=args.threshold)
