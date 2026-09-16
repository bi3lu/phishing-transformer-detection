"""Advanced analysis for phishing detection thesis.

Includes: error analysis, probability distribution comparison,
McNemar statistical significance tests, and ensemble ablation study.
"""

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import BASE_DIR, RESULTS_DIR, SAVED_MODELS_DIR, TEMPLATE_GROUP_COL
from src.data.load_data import load_split, prepare_xy
from src.evaluation.calibration import load_calibrator, resolve_project_path
from src.evaluation.inference import discover_models, predict_model
from src.evaluation.threshold_analysis import calculate_threshold_metrics, select_f1_threshold
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Main analysis:
def get_all_predictions(
    texts: List[str],
    X_series: pd.Series,
    models: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, NDArray[np.float64]]:
    """Get predictions from all available models.

    Returns dict mapping model name -> probability array.
    """
    if models is None:
        models = discover_models()

    predictions: Dict[str, NDArray[np.float64]] = {}

    for m in models:
        logger.info(f"Getting predictions from {m['name']}...")
        predictions[str(m["name"])] = predict_model(m, X_series)

    return predictions


def error_analysis(
    predictions: Dict[str, NDArray[np.float64]],
    y_true: NDArray[np.int_],
    texts: List[str],
    threshold: float = 0.5,
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Analyze misclassified samples across all models.

    Returns a DataFrame with each test sample, true label, each model's
    prediction and probability, and a count of how many models got it wrong.
    """
    n = len(y_true)
    df = pd.DataFrame({"text": texts, "true_label": y_true})

    for name, probs in predictions.items():
        df[f"{name}_prob"] = probs
        model_threshold = (thresholds or {}).get(name, threshold)
        df[f"{name}_pred"] = (probs >= model_threshold).astype(int)
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


def probability_distribution(
    predictions: Dict[str, NDArray[np.float64]],
    y_true: NDArray[np.int_],
    thresholds: Optional[Dict[str, float]] = None,
) -> None:
    """Plot descriptive calibrated-probability distributions by class."""
    import matplotlib.pyplot as plt

    n_models = len(predictions)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 4 * n_models), constrained_layout=True)

    if n_models == 1:
        axes = [axes]

    for ax, (name, probs) in zip(axes, predictions.items()):
        mask_legit = y_true == 0
        mask_phish = y_true == 1

        ax.hist(probs[mask_legit], bins=50, alpha=0.6, label="Legitimate", color="green", density=True)
        ax.hist(probs[mask_phish], bins=50, alpha=0.6, label="Phishing", color="red", density=True)
        model_threshold = (thresholds or {}).get(name, 0.5)
        ax.axvline(
            x=model_threshold,
            color="black",
            linestyle="--",
            label=f"Frozen validation threshold={model_threshold:.2f}",
        )
        ax.set_title(f"{name} — Probability Distribution")
        ax.set_xlabel("P(phishing)")
        ax.set_ylabel("Density")
        ax.legend()

    output_path = RESULTS_DIR / "probability_distributions.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Probability distributions saved to {output_path}")


def _holm_adjust(p_values: List[float]) -> List[float]:
    """Return Holm family-wise-error adjusted p-values."""
    count = len(p_values)
    adjusted = [1.0] * count
    running_max = 0.0

    for rank, index in enumerate(sorted(range(count), key=lambda item: p_values[item])):
        candidate = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max

    return adjusted


def mcnemar_test(
    predictions: Dict[str, NDArray[np.float64]],
    y_true: NDArray[np.int_],
    groups: NDArray[np.str_],
    threshold: float = 0.5,
    thresholds: Optional[Dict[str, float]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run exact pairwise McNemar tests on independent template families."""
    from scipy.stats import binomtest

    model_names = list(predictions.keys())
    group_values = np.asarray(groups, dtype=str)

    if not all(len(values) == len(y_true) == len(group_values) for values in predictions.values()):
        raise ValueError("Predictions, labels, and groups must have equal lengths")

    frame = pd.DataFrame({"group": group_values})

    for name, probabilities in predictions.items():
        frame[name] = (probabilities >= (thresholds or {}).get(name, threshold)).astype(int) == y_true

    grouped = frame.groupby("group", sort=True)[model_names].all()
    correct = {name: grouped[name].to_numpy(dtype=bool) for name in model_names}

    results: List[Dict[str, Any]] = []

    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1 :]:
            # b = A wrong, B right; c = A right, B wrong
            b = int(((~correct[name_a]) & correct[name_b]).sum())
            c = int((correct[name_a] & (~correct[name_b])).sum())

            discordant = b + c
            p_value = (
                float(binomtest(min(b, c), n=discordant, p=0.5, alternative="two-sided").pvalue) if discordant else 1.0
            )

            results.append(
                {
                    "model_a": name_a,
                    "model_b": name_b,
                    "b_a_wrong_b_right": b,
                    "c_a_right_b_wrong": c,
                    "discordant_template_groups": discordant,
                    "n_template_groups": len(grouped),
                    "p_value_exact": p_value,
                    "analysis_unit": "Template_Group_all_variants_correct",
                }
            )

    df = pd.DataFrame(results)

    if not df.empty:
        df["p_value_holm"] = _holm_adjust(df["p_value_exact"].astype(float).tolist())
        df["reject_holm"] = df["p_value_holm"] <= alpha
        df["alpha_familywise"] = alpha
        df["correction"] = "Holm"

    output_path = RESULTS_DIR / "mcnemar_tests.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"McNemar pairwise tests saved to {output_path}")
    logger.info(f"\n{df.to_string(index=False)}")

    return df


def _load_frozen_selection() -> Dict[str, Any]:
    path = RESULTS_DIR / "threshold_selection" / "thresholds.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing frozen threshold/calibration configuration: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = json.load(handle)

    if payload.get("selection_predictions") != "group_disjoint_calibration_oof":
        raise ValueError("Analysis requires group-disjoint OOF calibrated threshold selection")

    return payload


def _calibrate_predictions(
    predictions: Dict[str, NDArray[np.float64]], frozen: Dict[str, Any]
) -> Dict[str, NDArray[np.float64]]:
    calibrated: Dict[str, NDArray[np.float64]] = {}

    for name, probabilities in predictions.items():
        model_config = frozen.get("models", {}).get(name)

        if model_config is None:
            raise ValueError(f"Missing frozen calibration for {name}")

        calibrated[name] = load_calibrator(dict(model_config["calibrator"])).predict(probabilities)

    return calibrated


def _load_validation_oof_predictions(
    model_names: List[str], validation_df: pd.DataFrame, frozen: Dict[str, Any]
) -> Dict[str, NDArray[np.float64]]:
    from src.evaluation.ensemble import load_oof_validation_predictions

    return load_oof_validation_predictions([{"name": name} for name in model_names], validation_df, frozen)


def ensemble_ablation(
    predictions: Dict[str, NDArray[np.float64]],
    y_true: NDArray[np.int_],
) -> pd.DataFrame:
    """Test all 2+ model combinations to find the best ensemble.

    Uses equal weights and selects each threshold on validation data only.
    """
    model_names = list(predictions.keys())
    results = []

    for size in range(2, len(model_names) + 1):
        for combo in itertools.combinations(model_names, size):
            # Equal-weight averaging:
            avg_probs = np.mean([predictions[name] for name in combo], axis=0)
            selected = select_f1_threshold(calculate_threshold_metrics(y_true, avg_probs))
            auc = roc_auc_score(y_true, avg_probs)

            results.append(
                {
                    "models": " + ".join(combo),
                    "n_models": size,
                    "threshold": round(float(selected["threshold"]), 4),
                    "f1": round(float(selected["f1"]), 4),
                    "precision": round(float(selected["precision"]), 4),
                    "recall": round(float(selected["recall"]), 4),
                    "roc_auc": round(auc, 4),
                }
            )

    df = pd.DataFrame(results).sort_values("f1", ascending=False)

    output_path = RESULTS_DIR / "validation_ensemble_ablation.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Validation-only ensemble ablation ({len(df)} combinations) saved to {output_path}")
    logger.info(f"\nTop 10 combinations by F1:\n{df.head(10).to_string(index=False)}")

    return df


# Main:
def main() -> None:
    """Analyze cached, verified final predictions including the ensemble."""
    from src.evaluation.evaluate import load_final_predictions

    frame = load_final_predictions()
    predictions = {}
    thresholds = {}
    reference = None

    for name, rows in frame.groupby("model", sort=True):
        rows = rows.sort_values("Record_ID")
        predictions[str(name)] = rows.probability.to_numpy(dtype=float)
        thresholds[str(name)] = float(rows.threshold.iloc[0])
        reference = rows

    assert reference is not None

    labels = reference.Is_Phishing.to_numpy(dtype=int)
    errors = error_analysis(predictions, labels, reference.Text.tolist(), thresholds=thresholds)

    # error_analysis preserves the original position in the sorted output index:
    errors["Record_ID"] = reference.Record_ID.to_numpy()[errors.index.to_numpy()]
    errors.to_csv(RESULTS_DIR / "error_analysis.csv", index=False)
    probability_distribution(predictions, labels, thresholds=thresholds)
    mcnemar_test(predictions, labels, reference.Template_Group.to_numpy(dtype=str), thresholds=thresholds)
    validation = load_split("val")
    frozen = _load_frozen_selection()
    oof = _load_validation_oof_predictions(list(frozen["models"]), validation, frozen)
    ensemble_ablation(oof, validation.Is_Phishing.to_numpy(dtype=int))


if __name__ == "__main__":
    main()
