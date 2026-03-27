"""Advanced analysis for phishing detection thesis.

Includes: error analysis, probability distribution comparison,
McNemar statistical significance tests, and ensemble ablation study.
"""

import gc
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
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

SAVED_MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"

ALL_MODELS: List[Dict[str, str]] = [
    {"name": "baseline", "type": "sklearn"},
    {"name": "herbert-base", "type": "transformer"},
    {"name": "polish-roberta-v2", "type": "transformer"},
    {"name": "xlm-roberta-base", "type": "transformer"},
    {"name": "fine_tuned_bert", "type": "transformer"},
    {"name": "distilbert-multilingual", "type": "transformer"},
]


# ── Prediction helpers ──────────────────────────────────────────────


def _predict_transformer(
    model_path: str,
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 256,
) -> NDArray[np.floating[Any]]:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    all_probs: List[float] = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Inference ({Path(model_path).name})"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(
            device
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    del model, tokenizer
    
    gc.collect()
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return np.array(all_probs, dtype=np.float64)


def _predict_sklearn(model_path: str, texts: pd.Series) -> NDArray[np.floating[Any]]:
    pipeline = joblib.load(Path(model_path) / "pipeline.joblib")
    probs = pipeline.predict_proba(texts)[:, 1]
    return np.asarray(probs, dtype=np.float64)


def get_all_predictions(
    texts: List[str],
    X_series: pd.Series,
    models: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, NDArray[np.floating[Any]]]:
    """Get predictions from all available models.

    Returns dict mapping model name -> probability array.
    """
    if models is None:
        models = ALL_MODELS

    predictions: Dict[str, NDArray[np.floating[Any]]] = {}
    for m in models:
        model_path = str(SAVED_MODELS_DIR / m["name"])
        
        if not Path(model_path).exists():
            logger.warning(f"Model not found: {model_path}, skipping.")
            continue
        
        logger.info(f"Getting predictions from {m['name']}...")
        
        if m["type"] == "transformer":
            predictions[m["name"]] = _predict_transformer(model_path, texts)
            
        else:
            predictions[m["name"]] = _predict_sklearn(model_path, X_series)
            
    return predictions


# ── 1. Error Analysis ────────────────────────────────────────────────


def error_analysis(
    predictions: Dict[str, NDArray[np.floating[Any]]],
    y_true: NDArray[np.int_],
    texts: List[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Analyze misclassified samples across all models.

    Returns a DataFrame with each test sample, true label, each model's
    prediction and probability, and a count of how many models got it wrong.
    """
    n = len(y_true)
    df = pd.DataFrame({"text": texts, "true_label": y_true})

    for name, probs in predictions.items():
        df[f"{name}_prob"] = probs
        df[f"{name}_pred"] = (probs >= threshold).astype(int)
        df[f"{name}_correct"] = df[f"{name}_pred"] == df["true_label"]

    correct_cols = [c for c in df.columns if c.endswith("_correct")]
    df["models_wrong"] = len(correct_cols) - df[correct_cols].sum(axis=1)
    df["all_correct"] = df["models_wrong"] == 0

    # Sort by hardness (most models wrong first):
    df = df.sort_values("models_wrong", ascending=False)

    # Summary stats:
    n_all_correct = df["all_correct"].sum()
    n_any_wrong = n - n_all_correct
    logger.info(f"Error analysis: {n_all_correct}/{n} samples classified correctly by ALL models")
    logger.info(f"  {n_any_wrong} samples misclassified by at least one model")

    hardest = df[df["models_wrong"] == df["models_wrong"].max()]
    logger.info(f"  Hardest samples (wrong by {int(df['models_wrong'].max())} models): {len(hardest)}")

    output_path = RESULTS_DIR / "error_analysis.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Full error analysis saved to {output_path}")

    # Per-model error summary:
    summary_rows = []
    
    for name in predictions:
        pred_col = f"{name}_pred"
        correct_col = f"{name}_correct"
        wrong = (~df[correct_col]).sum()
        fp = ((df[pred_col] == 1) & (df["true_label"] == 0)).sum()
        fn = ((df[pred_col] == 0) & (df["true_label"] == 1)).sum()
        summary_rows.append({"model": name, "total_errors": wrong, "false_positives": fp, "false_negatives": fn})

    summary_df = pd.DataFrame(summary_rows)
    logger.info(f"\nPer-model error summary:\n{summary_df.to_string(index=False)}")

    summary_path = RESULTS_DIR / "error_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return df


# ── 2. Probability Distribution Comparison ──────────────────────────


def probability_distribution(
    predictions: Dict[str, NDArray[np.floating[Any]]],
    y_true: NDArray[np.int_],
) -> None:
    """Plot probability distribution histograms for each model, split by class."""
    n_models = len(predictions)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 4 * n_models), constrained_layout=True)
    
    if n_models == 1:
        axes = [axes]

    for ax, (name, probs) in zip(axes, predictions.items()):
        mask_legit = y_true == 0
        mask_phish = y_true == 1

        ax.hist(probs[mask_legit], bins=50, alpha=0.6, label="Legitimate", color="green", density=True)
        ax.hist(probs[mask_phish], bins=50, alpha=0.6, label="Phishing", color="red", density=True)
        ax.axvline(x=0.5, color="black", linestyle="--", label="Threshold=0.5")
        ax.set_title(f"{name} — Probability Distribution")
        ax.set_xlabel("P(phishing)")
        ax.set_ylabel("Density")
        ax.legend()

    output_path = RESULTS_DIR / "probability_distributions.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Probability distributions saved to {output_path}")


# ── 3. McNemar Test ──────────────────────────────────────────────────


def mcnemar_test(
    predictions: Dict[str, NDArray[np.floating[Any]]],
    y_true: NDArray[np.int_],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Run pairwise McNemar tests between all model pairs.

    McNemar test checks if two classifiers have statistically significantly
    different error rates. Uses the chi-squared approximation (with
    continuity correction) for speed and simplicity.

    Returns DataFrame with model_a, model_b, b (A wrong & B right),
    c (A right & B wrong), chi2, p_value, significant (p < 0.05).
    """
    from scipy.stats import chi2 as chi2_dist

    model_names = list(predictions.keys())
    preds = {name: (probs >= threshold).astype(int) for name, probs in predictions.items()}
    correct = {name: (preds[name] == y_true) for name in model_names}

    results = []
    
    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1 :]:
            # b = A wrong, B right; c = A right, B wrong
            b = int(((~correct[name_a]) & correct[name_b]).sum())
            c = int((correct[name_a] & (~correct[name_b])).sum())

            # McNemar chi-squared with continuity correction:
            if b + c == 0:
                chi2 = 0.0
                p_value = 1.0
                
            else:
                chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                p_value = 1 - chi2_dist.cdf(chi2, df=1)

            results.append(
                {
                    "model_a": name_a,
                    "model_b": name_b,
                    "b_a_wrong_b_right": b,
                    "c_a_right_b_wrong": c,
                    "chi2": round(chi2, 4),
                    "p_value": round(p_value, 6),
                    "significant": p_value < 0.05,
                }
            )

    df = pd.DataFrame(results)
    output_path = RESULTS_DIR / "mcnemar_tests.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"McNemar pairwise tests saved to {output_path}")
    logger.info(f"\n{df.to_string(index=False)}")
    
    return df


# ── 4. Ensemble Ablation Study ───────────────────────────────────────


def ensemble_ablation(
    predictions: Dict[str, NDArray[np.floating[Any]]],
    y_true: NDArray[np.int_],
    threshold: float = 0.35,
) -> pd.DataFrame:
    """Test all 2+ model combinations to find the best ensemble.

    Uses equal weights for simplicity. Reports F1, precision, recall,
    ROC-AUC for each combination.
    """
    model_names = list(predictions.keys())
    results = []

    for size in range(2, len(model_names) + 1):
        for combo in itertools.combinations(model_names, size):
            # Equal-weight averaging:
            avg_probs = np.mean([predictions[name] for name in combo], axis=0)
            preds = (avg_probs >= threshold).astype(int)

            f1 = f1_score(y_true, preds)
            precision = precision_score(y_true, preds)
            recall = recall_score(y_true, preds)
            auc = roc_auc_score(y_true, avg_probs)

            results.append(
                {
                    "models": " + ".join(combo),
                    "n_models": size,
                    "f1": round(f1, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "roc_auc": round(auc, 4),
                }
            )

    df = pd.DataFrame(results).sort_values("f1", ascending=False)

    output_path = RESULTS_DIR / "ensemble_ablation.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Ensemble ablation ({len(df)} combinations) saved to {output_path}")
    logger.info(f"\nTop 10 combinations by F1:\n{df.head(10).to_string(index=False)}")
    
    return df


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Run all analyses: error, probability distributions, McNemar, ablation."""
    logger.info("Loading test data...")
    df_test = load_split("test")
    X_test, y_test = prepare_xy(df_test)
    texts = X_test.tolist()
    y_arr = y_test.values

    logger.info("Collecting predictions from all models...")
    predictions = get_all_predictions(texts, X_test)

    logger.info("\n" + "=" * 60)
    logger.info("ERROR ANALYSIS")
    logger.info("=" * 60)
    error_analysis(predictions, y_arr, texts)

    logger.info("\n" + "=" * 60)
    logger.info("PROBABILITY DISTRIBUTIONS")
    logger.info("=" * 60)
    probability_distribution(predictions, y_arr)

    logger.info("\n" + "=" * 60)
    logger.info("McNEMAR STATISTICAL SIGNIFICANCE TESTS")
    logger.info("=" * 60)
    mcnemar_test(predictions, y_arr)

    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE ABLATION STUDY")
    logger.info("=" * 60)
    ensemble_ablation(predictions, y_arr)

    logger.info("\nAll analyses complete.")


if __name__ == "__main__":
    main()
