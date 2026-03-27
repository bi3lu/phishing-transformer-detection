"""Ensemble inference for phishing detection.

Combines predictions from multiple models (transformers + baseline)
using weighted probability averaging to produce a stronger classifier.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import BASE_DIR, LABEL_COL, TEXT_COL
from src.data.load_data import load_split, prepare_xy
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default ensemble configuration — weights reflect model quality:
DEFAULT_ENSEMBLE: List[Dict[str, Any]] = [
    {"name": "herbert-base", "type": "transformer", "weight": 0.25},
    {"name": "polish-roberta-v2", "type": "transformer", "weight": 0.20},
    {"name": "xlm-roberta-base", "type": "transformer", "weight": 0.15},
    {"name": "fine_tuned_bert", "type": "transformer", "weight": 0.20},
    {"name": "baseline", "type": "sklearn", "weight": 0.20},
]


def _predict_transformer(
    model_path: str,
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 512,
) -> NDArray[np.floating[Any]]:
    """Get positive-class probabilities from a transformer model."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    all_probs: List[float] = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Inference ({Path(model_path).name})"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    return np.array(all_probs, dtype=np.float64)


def _predict_sklearn(model_path: str, texts: pd.Series) -> NDArray[np.floating[Any]]:
    """Get positive-class probabilities from a sklearn pipeline."""
    pipeline = joblib.load(Path(model_path) / "pipeline.joblib")
    probs = pipeline.predict_proba(texts)[:, 1]
    return np.asarray(probs, dtype=np.float64)


def ensemble_predict(
    texts: List[str],
    X_series: pd.Series,
    ensemble_config: Optional[List[Dict[str, Any]]] = None,
    saved_models_dir: Optional[Path] = None,
) -> NDArray[np.floating[Any]]:
    """Generate ensemble predictions by weighted averaging.

    Args:
        texts: List of text strings for transformer models.
        X_series: Pandas Series of texts for sklearn models.
        ensemble_config: List of model configs with name, type, weight.
        saved_models_dir: Directory containing saved models.

    Returns:
        Weighted average of positive-class probabilities.
    """
    if ensemble_config is None:
        ensemble_config = DEFAULT_ENSEMBLE

    if saved_models_dir is None:
        saved_models_dir = BASE_DIR / "saved_models"

    weighted_probs = np.zeros(len(texts), dtype=np.float64)
    total_weight = 0.0

    for model_cfg in ensemble_config:
        name = model_cfg["name"]
        model_type = model_cfg["type"]
        weight = float(model_cfg["weight"])
        model_path = str(saved_models_dir / name)

        if not Path(model_path).exists():
            logger.warning(f"Model not found: {model_path}, skipping.")
            continue

        logger.info(f"Loading {name} (weight={weight})...")

        if model_type == "transformer":
            probs = _predict_transformer(model_path, texts)

        elif model_type == "sklearn":
            probs = _predict_sklearn(model_path, X_series)

        else:
            logger.error(f"Unknown model type: {model_type}")
            continue

        weighted_probs += weight * probs
        total_weight += weight

    if total_weight == 0:
        raise RuntimeError("No models were successfully loaded for ensemble.")

    # Normalize by actual weight used (in case some models were skipped):
    return weighted_probs / total_weight


def main(threshold: float = 0.5) -> None:
    """Run ensemble evaluation on the test set.

    Args:
        threshold: Classification threshold. If 0.0, parsed from CLI.
    """
    if threshold == 0.0:
        parser = argparse.ArgumentParser(description="Ensemble inference for phishing detection.")
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.5,
            help="Classification threshold (default: 0.5)",
        )
        args = parser.parse_args()
        threshold = args.threshold

    logger.info("Loading test data...")
    df_test = load_split("test")
    X_test, y_test = prepare_xy(df_test)

    texts = X_test.tolist()

    logger.info("Running ensemble inference...")
    ensemble_probs = ensemble_predict(texts, X_test)

    # Evaluate at given threshold:
    preds = (ensemble_probs >= threshold).astype(int)

    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    auc = roc_auc_score(y_test, ensemble_probs)

    logger.info("=== Ensemble Results ===")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"F1:        {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"ROC-AUC:   {auc:.4f}")
    logger.info("\n" + classification_report(y_test, preds))

    # Save results:
    output_path = BASE_DIR / "results" / "ensemble_results.csv"
    pd.DataFrame([{"threshold": threshold, "f1": f1, "precision": precision, "recall": recall, "roc_auc": auc}]).to_csv(
        output_path, index=False
    )
    logger.info(f"Results saved to {output_path}")

    # Also save per-sample probabilities for threshold analysis:
    probs_df = pd.DataFrame({"text": texts, "true_label": y_test, "ensemble_prob": ensemble_probs})
    probs_path = BASE_DIR / "results" / "ensemble_probabilities.csv"
    probs_df.to_csv(probs_path, index=False)
    logger.info(f"Per-sample probabilities saved to {probs_path}")


if __name__ == "__main__":
    main(threshold=0.0)
