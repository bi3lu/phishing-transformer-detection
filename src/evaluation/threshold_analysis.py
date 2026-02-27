import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.config import BASE_DIR
from src.data.load_data import load_split, prepare_xy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_threshold_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    fp_cost: float = 1.0,
    fn_cost: float = 20.0,
) -> pd.DataFrame:
    """
    Calculate metrics for a range of thresholds.

    Args:
        y_true: True labels (0 or 1).
        y_probs: Predicted probabilities for the positive class (1).
        thresholds: Array of thresholds to evaluate. If None, uses np.linspace(0.05, 0.95, 19).
        fp_cost: Cost of a False Positive.
        fn_cost: Cost of a False Negative.

    Returns:
        pd.DataFrame containing metrics for each threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)  # 0.05 step

    results = []

    for thr in thresholds:
        y_pred = (y_probs >= thr).astype(int)

        # Calculate confusion matrix components:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Calculate metrics:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # False positive rate:
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Business cost:
        cost = (fp * fp_cost) + (fn * fn_cost)

        results.append(
            {
                "threshold": thr,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "cost": cost,
            }
        )

    return pd.DataFrame(results)


def load_predictions_sklearn(model_path: Union[str, Path], X: pd.Series) -> np.ndarray:  # TODO: Add docstring
    logger.info(f"Loading sklearn model from {model_path}...")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    probs = model.predict_proba(X)[:, 1]

    return probs


def load_predictions_transformer(
    model_path: Union[str, Path],
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:  # TODO: Add docstring
    logger.info(f"Loading transformer model from {model_path}...")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    all_probs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
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
            class_1_probs = probs[:, 1].cpu().numpy()
            all_probs.extend(class_1_probs)

    return np.array(all_probs)


def plot_metrics(df_results: pd.DataFrame, model_name: str, output_dir: Path) -> None:  # TODO: Add docstring
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Precision-Recall vs Threshold:
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.figure(figsize=(10, 6))
    plt.plot(df_results["threshold"], df_results["precision"], label="Precision", marker=".")

    plt.plot(df_results["threshold"], df_results["recall"], label="Recall", marker=".")
    plt.plot(
        df_results["threshold"],
        df_results["f1"],
        label="F1 Score",
        marker=".",
        linestyle="--",
    )

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision, Recall, F1 vs Threshold - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f"pr_recall_f1_{model_name}_{timestamp}.png")
    plt.close()

    # 2. Cost vs Threshold:
    plt.figure(figsize=(10, 6))
    plt.plot(
        df_results["threshold"],
        df_results["cost"],
        label="Cost",
        color="red",
        marker="o",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Total Cost")
    plt.title(f"Cost vs Threshold - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f"cost_curve_{model_name}_{timestamp}.png")
    plt.close()

    # 3. Precision-Recall Curve:
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["recall"], df_results["precision"], marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.grid(True)
    plt.savefig(output_dir / f"pr_curve_{model_name}_{timestamp}.png")
    plt.close()


# Main:
def main() -> None:
    output_dir = BASE_DIR / "results" / "threshold_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Test Data:
    logger.info("Loading test data...")
    df_test = load_split("test")
    X_test, y_test = prepare_xy(df_test)

    # Baseline:
    models_config: List[Dict[str, Any]] = [
        {
            "name": "Baseline_LR",
            "type": "sklearn",
            "path": BASE_DIR / "mlruns/1/models/m-10b0ac281d4c40629b89914e7f92dbb0/artifacts/model.pkl",
        }
    ]

    # Fine-tunded transformers:
    saved_models_dir = BASE_DIR / "saved_models"

    if saved_models_dir.exists():
        for model_dir in saved_models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                models_config.append({"name": model_dir.name, "type": "transformer", "path": model_dir})

    summary_metrics = []

    for config in models_config:
        model_name = str(config["name"])
        model_path = Path(str(config["path"]))
        model_type = str(config["type"])

        if not os.path.exists(model_path):
            logger.warning(f"Model path not found: {model_path}. Skipping {model_name}.")
            continue

        logger.info(f"Processing {model_name}...")

        # Get predictions:
        if model_type == "sklearn":
            y_probs = load_predictions_sklearn(model_path, X_test)

        elif model_type == "transformer":
            y_probs = load_predictions_transformer(model_path, X_test.tolist())

        else:
            logger.error(f"Unknown model type: {model_type}")
            continue

        # Calculate metrics:
        df_results = calculate_threshold_metrics(y_test, y_probs)

        # Save full results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"threshold_analysis_{model_name}_{timestamp}.csv"
        df_results.to_csv(csv_path, index=False)
        logger.info(f"Saved threshold analysis to {csv_path}")

        # Plotting:
        plot_metrics(df_results, model_name, output_dir)

        # Select specific thresholds for report:

        # 1. Best F1:
        best_f1_idx = df_results["f1"].idxmax()
        best_f1_row = df_results.loc[best_f1_idx]

        # 2. Lowest Cost:
        min_cost_idx = df_results["cost"].idxmin()
        min_cost_row = df_results.loc[min_cost_idx]

        # 3. High Precision:
        low_fpr_rows = df_results[df_results["fpr"] < 0.01]
        if not low_fpr_rows.empty:
            high_prec_row = low_fpr_rows.sort_values("recall", ascending=False).iloc[0]

        else:
            high_prec_row = df_results.sort_values("fpr").iloc[0]

        # 4. Default 0.5:
        default_idx = (df_results["threshold"] - 0.5).abs().idxmin()
        default_row = df_results.loc[default_idx]

        # Print report:
        print(f"\nMODEL: {model_name}")
        print(
            f"Best F1 (Thr={best_f1_row['threshold']:.2f}): F1={best_f1_row['f1']:.4f}, Cost={best_f1_row['cost']:.1f}"
        )
        print(
            f"Min Cost (Thr={min_cost_row['threshold']:.2f}): Cost={min_cost_row['cost']:.1f}, F1={min_cost_row['f1']:.4f}"
        )
        print(
            f"High Precision (Thr={high_prec_row['threshold']:.2f}): Precision={high_prec_row['precision']:.4f}, FPR={high_prec_row['fpr']:.4f}, Recall={high_prec_row['recall']:.4f}"
        )
        print(
            f"Default 0.5 (Thr={default_row['threshold']:.2f}): F1={default_row['f1']:.4f}, Cost={default_row['cost']:.1f}"
        )

        # Collect for summary:
        summary_metrics.append(
            {
                "model": model_name,
                "best_f1": best_f1_row["f1"],
                "min_cost": min_cost_row["cost"],
                "high_prec_f1": high_prec_row["f1"],
            }
        )


# Entry point:
if __name__ == "__main__":
    main()
