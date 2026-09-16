"""Validation-selected equal-weight ensemble with one final test evaluation."""

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score

from src.config import BASE_DIR, DEFAULT_MAX_LENGTH, LABEL_COL, RESULTS_DIR, SOURCE_COL, TEMPLATE_GROUP_COL, TEXT_COL
from src.data.load_data import load_split, prepare_xy
from src.evaluation.calibration import calibration_metrics, file_sha256, load_calibrator, resolve_project_path
from src.evaluation.inference import discover_models, predict_model
from src.evaluation.threshold_analysis import calculate_threshold_metrics, select_f1_threshold
from src.utils.logger import get_logger

logger = get_logger(__name__)


def average_probabilities(
    predictions: Dict[str, NDArray[np.float64]], model_names: Iterable[str]
) -> NDArray[np.float64]:
    """Average model probabilities with preregistered equal weights."""
    names = list(model_names)

    if not names:
        raise ValueError("At least one model is required")

    return np.asarray(np.mean([predictions[name] for name in names], axis=0), dtype=np.float64)


def collect_predictions(models: List[Dict[str, Any]], texts: pd.Series) -> Dict[str, NDArray[np.float64]]:
    """Run each model once for a given split."""
    return {str(model["name"]): predict_model(model, texts) for model in models}


def load_oof_validation_predictions(
    models: List[Dict[str, Any]], validation_df: pd.DataFrame, frozen: Dict[str, Any]
) -> Dict[str, NDArray[np.float64]]:
    """Load group-disjoint calibrated OOF predictions selected upstream."""
    predictions: Dict[str, NDArray[np.float64]] = {}

    for model in models:
        name = str(model["name"])
        model_config = frozen.get("models", {}).get(name)

        if model_config is None:
            raise ValueError(f"Missing validation calibration for ensemble model: {name}")

        load_calibrator(dict(model_config["calibrator"]))
        path = resolve_project_path(str(model_config["validation_oof_probabilities_path"]))

        if file_sha256(path) != model_config["validation_oof_probabilities_sha256"]:
            raise ValueError(f"OOF probabilities content changed: {name}")

        frame = pd.read_csv(path)

        if not frame["Record_ID"].equals(validation_df["Record_ID"]):
            raise ValueError(f"OOF record order changed: {name}")

        if len(frame) != len(validation_df):
            raise ValueError(f"Stale OOF validation probabilities for {name}")

        if not frame[TEMPLATE_GROUP_COL].astype(str).equals(validation_df[TEMPLATE_GROUP_COL].astype(str)):
            raise ValueError(f"OOF validation group order changed for {name}")

        if not frame[LABEL_COL].astype(int).equals(validation_df[LABEL_COL].astype(int)):
            raise ValueError(f"OOF validation labels changed for {name}")

        predictions[name] = frame["calibrated_oof_probability"].to_numpy(dtype=np.float64)

    return predictions


def calibrate_predictions(
    predictions: Dict[str, NDArray[np.float64]], frozen: Dict[str, Any]
) -> Dict[str, NDArray[np.float64]]:
    """Apply validation-fitted final calibrators to a new split."""
    result: Dict[str, NDArray[np.float64]] = {}

    for name, probabilities in predictions.items():
        model_config = frozen.get("models", {}).get(name)

        if model_config is None:
            raise ValueError(f"Missing frozen calibrator for ensemble model: {name}")

        result[name] = load_calibrator(dict(model_config["calibrator"])).predict(probabilities)

    return result


def select_ensemble(
    predictions: Dict[str, NDArray[np.float64]], y_validation: NDArray[np.int_]
) -> tuple[Dict[str, Any], pd.DataFrame]:
    """Select members and threshold using validation data only."""
    names = sorted(predictions)

    if len(names) < 2:
        raise ValueError("At least two complete models are required for ensemble selection")

    candidates = []

    for size in range(2, len(names) + 1):
        for combination in itertools.combinations(names, size):
            probabilities = average_probabilities(predictions, combination)
            threshold_row = select_f1_threshold(calculate_threshold_metrics(y_validation, probabilities))
            probability_metrics = calibration_metrics(y_validation, probabilities)
            candidates.append(
                {
                    "models": list(combination),
                    "n_models": size,
                    "threshold": float(threshold_row["threshold"]),
                    "f1": float(threshold_row["f1"]),
                    "precision": float(threshold_row["precision"]),
                    "recall": float(threshold_row["recall"]),
                    "roc_auc": float(roc_auc_score(y_validation, probabilities)),
                    "brier_score": probability_metrics["brier_score"],
                    "ece": probability_metrics["ece"],
                }
            )

    table = pd.DataFrame(candidates)
    table["models_display"] = table["models"].map(" + ".join)
    table.sort_values(["f1", "n_models", "roc_auc"], ascending=[False, True, False], inplace=True)
    best = table.iloc[0]
    config = {
        "selection_split": "validation",
        "selection_objective": "max_f1_then_fewer_models_then_roc_auc",
        "weighting": "equal",
        "probability_inputs": "per_model_group_disjoint_oof_calibrated",
        "models": list(best["models"]),
        "threshold": float(best["threshold"]),
        "validation_f1": float(best["f1"]),
        "validation_brier_score": float(best["brier_score"]),
        "validation_ece": float(best["ece"]),
        "max_length": DEFAULT_MAX_LENGTH,
    }
    return config, table.drop(columns=["models"])


def evaluate_frozen_ensemble(
    config: Dict[str, Any],
    predictions: Dict[str, NDArray[np.float64]],
    y_test: NDArray[np.int_],
) -> tuple[Dict[str, float], NDArray[np.float64]]:
    """Evaluate a previously selected configuration without further tuning."""
    probabilities = average_probabilities(predictions, config["models"])
    threshold = float(config["threshold"])
    predicted = (probabilities >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "f1": float(f1_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        **calibration_metrics(y_test, probabilities),
    }
    logger.info("=== Frozen Ensemble Final Test ===")
    logger.info(f"Models: {config['models']}")
    logger.info(f"Threshold selected on validation: {threshold:.2f}")
    logger.info("\n" + classification_report(y_test, predicted))
    return metrics, probabilities


def main() -> None:
    """Select on validation, freeze the choice, then evaluate test exactly once."""
    results_dir = RESULTS_DIR
    selection_dir = results_dir / "ensemble_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    models = discover_models()

    threshold_path = results_dir / "threshold_selection" / "thresholds.json"

    if not threshold_path.exists():
        raise FileNotFoundError("Run validation threshold/calibration selection before ensemble selection")

    with threshold_path.open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)

    if frozen.get("selection_predictions") != "group_disjoint_calibration_oof":
        raise ValueError("Ensemble requires group-disjoint OOF calibrated validation probabilities")

    # Phase 1: model/threshold selection. Test data is intentionally not loaded.
    validation_df = load_split("val")
    _, y_validation = prepare_xy(validation_df)
    validation_predictions = load_oof_validation_predictions(models, validation_df, frozen)
    config, ablation = select_ensemble(validation_predictions, y_validation.to_numpy())
    ablation.to_csv(selection_dir / "validation_ensemble_ablation.csv", index=False)
    config["threshold_configuration_sha256"] = file_sha256(threshold_path)
    config_path = selection_dir / "ensemble_config.json"

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    logger.info(f"Ensemble configuration frozen from validation data: {config_path}")


if __name__ == "__main__":
    main()
