import logging
from pathlib import Path
from typing import Tuple

import mlflow  # type: ignore
import pandas as pd  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import f1_score  # type: ignore
from sklearn.metrics import (classification_report, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline  # type: ignore

from src.config import LABEL_COL, RANDOM_STATE, SPLIT_DATA_DIR, TEXT_COL
from src.utils.logger import get_logger

# Setup logging:
logger = get_logger(__name__)


# Helper functions:
def _load_split(name: str) -> pd.DataFrame:  # TODO: Add docstring
    path = SPLIT_DATA_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    return pd.read_csv(path)


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:  # TODO: Add docstring
    X = df[TEXT_COL].astype(str)
    y = df[LABEL_COL].map({False: 0, True: 1}).astype(int)

    return X, y


# Main:
def main() -> None:
    logger.info("Running baseline TF-IDF + LogisticRegression")

    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")

    X_train, y_train = _prepare_xy(train_df)
    X_val, y_val = _prepare_xy(val_df)
    X_test, y_test = _prepare_xy(test_df)

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=50_000,
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    mlflow.set_experiment("phishing_baseline")

    with mlflow.start_run(run_name="tfidf_logreg"):
        logger.info("Training baseline model...")
        pipeline.fit(X_train, y_train)

        logger.info("Evaluating on validation set...")
        val_preds = pipeline.predict(X_val)
        val_probs = pipeline.predict_proba(X_val)[:, 1]

        val_f1 = f1_score(y_val, val_preds)
        val_precision = precision_score(y_val, val_preds)
        val_recall = recall_score(y_val, val_preds)
        val_auc = roc_auc_score(y_val, val_probs)

        logger.info("Validation results:")
        logger.info(f"F1: \t{val_f1:.4f}")
        logger.info(f"Precision: \t{val_precision:.4f}")
        logger.info(f"Recall: \t{val_recall:.4f}")
        logger.info(f"ROC-AUC: \t{val_auc:.4f}")

        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "vectorizer": "TF-IDF",
                "ngram_range": "(1,2)",
                "max_features": 50_000,
                "class_weight": "balanced",
            }
        )

        mlflow.log_metrics(
            {
                "val_f1": val_f1,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_auc": val_auc,
            }
        )

        # Final test evaluation:
        logger.info("Evaluating on test set...")
        test_preds = pipeline.predict(X_test)
        test_probs = pipeline.predict_proba(X_test)[:, 1]

        test_f1 = f1_score(y_test, test_preds)
        test_precision = precision_score(y_test, test_preds)
        test_recall = recall_score(y_test, test_preds)
        test_auc = roc_auc_score(y_test, test_probs)

        mlflow.log_metrics(
            {
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_auc": test_auc,
            }
        )

        logger.info("Test results:")
        logger.info(f"F1: \t{test_f1:.4f}")
        logger.info(f"Precision: \t{test_precision:.4f}")
        logger.info(f"Recall: \t{test_recall:.4f}")
        logger.info(f"ROC-AUC: \t{test_auc:.4f}")

        logger.info("Classification report (TEST):")
        logger.info("\n" + classification_report(y_test, test_preds))

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    logger.info("Baseline experiment finished successfully.")


# Entry point:
if __name__ == "__main__":
    main()
