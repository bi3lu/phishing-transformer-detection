import mlflow  # type: ignore
import yaml  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import f1_score  # type: ignore
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # type: ignore

from src.config import BASE_DIR, RANDOM_STATE
from src.data.load_data import load_split, prepare_xy
from src.utils.logger import get_logger

# Setup logging:
logger = get_logger(__name__)


# Logging metrics:
def log_metrics(y_true, y_pred, y_probs, prefix: str = "val") -> None:
    """Calculate and log metrics for a given split."""
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
    """Run the baseline phishing detection experiment.

    This function trains a text classification pipeline consisting of
    TF-IDF vectorization followed by Logistic Regression. It evaluates
    the model on validation and test splits, logs metrics and parameters
    to MLflow, saves the trained model artifact, and outputs detailed
    logs including a classification report.

    The dataset splits are expected to be stored as CSV files in
    ``SPLIT_DATA_DIR`` with names: train.csv, val.csv, and test.csv.
    """
    logger.info("Running baseline TF-IDF + LogisticRegression")

    # Load parameters from yaml config
    params_path = BASE_DIR / "params.yaml"
    with open(params_path, "r") as f:
        config = yaml.safe_load(f)

    tfidf_params = config["baseline"]["tfidf"]
    lr_params = config["baseline"]["logistic_regression"]

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)
    X_test, y_test = prepare_xy(test_df)

    pipeline = Pipeline(
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

    mlflow.set_experiment("phishing_baseline")

    with mlflow.start_run(run_name="tfidf_logreg"):
        logger.info("Training baseline model...")
        pipeline.fit(X_train, y_train)

        logger.info("Evaluating on validation set...")
        val_preds = pipeline.predict(X_val)
        val_probs = pipeline.predict_proba(X_val)[:, 1]

        log_metrics(y_val, val_preds, val_probs, prefix="val")

        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "vectorizer": "TF-IDF",
                "ngram_range": str(tuple(tfidf_params["ngram_range"])),
                "max_features": tfidf_params["max_features"],
                "class_weight": lr_params["class_weight"],
            }
        )

        # Final test evaluation:
        logger.info("Evaluating on test set...")
        test_preds = pipeline.predict(X_test)
        test_probs = pipeline.predict_proba(X_test)[:, 1]

        log_metrics(y_test, test_preds, test_probs, prefix="test")

        logger.info("Classification report (TEST):")
        logger.info("\n" + classification_report(y_test, test_preds))

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    logger.info("Baseline experiment finished successfully.")


# Entry point:
if __name__ == "__main__":
    main()
