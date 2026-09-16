"""Decision-threshold selection performed exclusively on validation data."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.config import BASE_DIR, DEFAULT_MAX_LENGTH, LABEL_COL, RESULTS_DIR, TEMPLATE_GROUP_COL
from src.data.load_data import load_split, prepare_xy
from src.evaluation.calibration import (
    CALIBRATION_METHOD,
    calibration_metrics,
    cross_fitted_calibration,
    plot_reliability_diagram,
    reliability_table,
    save_calibrator,
)
from src.evaluation.inference import discover_models, predict_model, predict_transformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_predictions_transformer(
    model_path: str,
    texts: list[str],
    batch_size: int = 16,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: Optional[str] = None,
) -> NDArray[np.float64]:
    return predict_transformer(Path(model_path), texts, batch_size, max_length, device)


def calculate_threshold_metrics(
    y_true: NDArray[np.int_],
    y_probs: NDArray[np.floating[Any]],
    thresholds: Optional[NDArray[np.floating[Any]]] = None,
) -> pd.DataFrame:
    """Calculate observed validation metrics for candidate thresholds."""
    if thresholds is None:
        unique = np.unique(np.asarray(y_probs, dtype=float))

        if unique.size == 0 or not np.isfinite(unique).all() or unique.min() < 0 or unique.max() > 1:
            raise ValueError("Probabilities must be finite and within [0, 1]")

        thresholds = np.unique(np.concatenate(([0.0, 0.5], unique, np.nextafter(unique, np.inf))))

    rows = []

    for threshold in thresholds:
        predictions = (y_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
                "fpr": fp / (fp + tn) if fp + tn else 0.0,
                "fnr": fn / (fn + tp) if fn + tp else 0.0,
                "tpr": tp / (tp + fn) if tp + fn else 0.0,
                "tnr": tn / (tn + fp) if tn + fp else 0.0,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )
    return pd.DataFrame(rows)


def calculate_deployment_scenarios(metrics: pd.DataFrame, scenarios: list[Dict[str, Any]]) -> pd.DataFrame:
    """Project precision and expected cost under explicit sensitivity scenarios.

    These rows are descriptive and never participate in canonical threshold
    selection. Costs are relative units, not empirically estimated currency.
    """
    rows = []
    for scenario in scenarios:
        prevalence = float(scenario["prevalence"])
        fp_cost = float(scenario["fp_cost"])
        fn_cost = float(scenario["fn_cost"])

        if not 0.0 < prevalence < 1.0:
            raise ValueError(f"Scenario prevalence must be between 0 and 1: {scenario}")

        if fp_cost < 0.0 or fn_cost < 0.0:
            raise ValueError(f"Scenario costs must be non-negative: {scenario}")

        for _, metric in metrics.iterrows():
            denominator = prevalence * metric["tpr"] + (1.0 - prevalence) * metric["fpr"]
            deployment_precision = prevalence * metric["tpr"] / denominator if denominator else 0.0
            expected_cost = (1.0 - prevalence) * metric["fpr"] * fp_cost + prevalence * metric["fnr"] * fn_cost
            rows.append(
                {
                    "scenario": str(scenario["name"]),
                    "assumption_status": "illustrative_sensitivity_not_empirically_estimated",
                    "threshold": float(metric["threshold"]),
                    "assumed_prevalence": prevalence,
                    "fp_cost_relative": fp_cost,
                    "fn_cost_relative": fn_cost,
                    "deployment_precision": float(deployment_precision),
                    "expected_cost_per_message": float(expected_cost),
                    "tpr_from_validation": float(metric["tpr"]),
                    "fpr_from_validation": float(metric["fpr"]),
                }
            )

    return pd.DataFrame(rows)


def select_f1_threshold(metrics: pd.DataFrame) -> pd.Series:
    """Select validation F1 optimum with a deterministic conservative tie-break."""
    if metrics.empty:
        raise ValueError("Cannot select a threshold from an empty metric table")

    ranked = metrics.assign(distance_from_default=(metrics["threshold"] - 0.5).abs()).sort_values(
        ["f1", "distance_from_default", "threshold"], ascending=[False, True, True]
    )

    return ranked.iloc[0]


def plot_validation_metrics(metrics: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    """Save a plot explicitly labelled as validation-only selection."""
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(10, 6))

    for metric in ("precision", "recall", "f1"):
        axis.plot(metrics["threshold"], metrics[metric], label=metric.capitalize())

    axis.set(xlabel="Threshold", ylabel="Score", title=f"Validation threshold selection — {model_name}")
    axis.legend()
    fig.savefig(output_dir / f"validation_threshold_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Select and freeze one threshold per model without reading test.csv."""
    from src.utils.artifacts import bind_run, file_sha256

    bind_run()
    output_dir = RESULTS_DIR / "threshold_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_df = load_split("val")
    x_validation, y_validation = prepare_xy(validation_df)

    if TEMPLATE_GROUP_COL not in validation_df.columns:
        raise ValueError(f"Validation data is missing {TEMPLATE_GROUP_COL}")

    with (BASE_DIR / "params.yaml").open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle).get("research_protocol", {})

    calibration_config = protocol.get("calibration", {})

    if calibration_config.get("method") != CALIBRATION_METHOD:
        raise ValueError(f"Canonical calibration method must be {CALIBRATION_METHOD}")

    calibration_folds = int(calibration_config.get("folds", 5))
    ece_bins = int(calibration_config.get("ece_bins", 10))
    deployment_scenarios = list(protocol.get("deployment_scenarios", []))

    if not deployment_scenarios:
        raise ValueError("At least one preregistered deployment sensitivity scenario is required")

    models = discover_models()

    if not models:
        raise RuntimeError("No complete registered models are available for threshold selection")

    selections: Dict[str, Any] = {
        "selection_split": "validation",
        "selection_predictions": "group_disjoint_calibration_oof",
        "objective": "max_f1",
        "max_length": DEFAULT_MAX_LENGTH,
        "calibration": {
            "method": CALIBRATION_METHOD,
            "folds": calibration_folds,
            "ece_bins": ece_bins,
            "fit_split": "validation",
        },
        "deployment_scenarios_are_descriptive_only": True,
        "models": {},
    }

    calibration_rows = []

    for model in models:
        name = str(model["name"])
        logger.info(f"Selecting threshold for {name} on validation data...")
        raw_probabilities = predict_model(model, x_validation)
        calibrated_probabilities, calibrator, fold_ids = cross_fitted_calibration(
            raw_probabilities,
            y_validation.to_numpy(),
            validation_df[TEMPLATE_GROUP_COL].astype(str).to_numpy(),
            n_splits=calibration_folds,
        )
        metrics = calculate_threshold_metrics(y_validation.to_numpy(), calibrated_probabilities)
        selected = select_f1_threshold(metrics)
        metrics.to_csv(output_dir / f"validation_thresholds_{name}.csv", index=False)
        scenario_table = calculate_deployment_scenarios(metrics, deployment_scenarios)
        scenario_table.to_csv(output_dir / f"validation_deployment_scenarios_{name}.csv", index=False)

        raw_calibration = calibration_metrics(y_validation.to_numpy(), raw_probabilities, n_bins=ece_bins)
        oof_calibration = calibration_metrics(y_validation.to_numpy(), calibrated_probabilities, n_bins=ece_bins)
        raw_reliability = reliability_table(y_validation.to_numpy(), raw_probabilities, n_bins=ece_bins).assign(
            model=name, probability_type="raw"
        )
        calibrated_reliability = reliability_table(
            y_validation.to_numpy(), calibrated_probabilities, n_bins=ece_bins
        ).assign(model=name, probability_type="calibrated_oof")
        pd.concat([raw_reliability, calibrated_reliability], ignore_index=True).to_csv(
            output_dir / f"validation_reliability_{name}.csv", index=False
        )
        plot_reliability_diagram(
            y_validation.to_numpy(),
            raw_probabilities,
            calibrated_probabilities,
            name,
            output_dir / f"validation_reliability_{name}.svg",
            n_bins=ece_bins,
        )
        calibrator_metadata = save_calibrator(
            calibrator,
            output_dir / "calibrators" / f"{name}.joblib",
            name,
            calibration_folds,
        )
        probability_path = output_dir / f"validation_oof_probabilities_{name}.csv"
        pd.DataFrame(
            {
                "Record_ID": validation_df["Record_ID"],
                "row_position": np.arange(len(validation_df)),
                TEMPLATE_GROUP_COL: validation_df[TEMPLATE_GROUP_COL].astype(str),
                LABEL_COL: y_validation.to_numpy(),
                "calibration_fold": fold_ids,
                "raw_probability": raw_probabilities,
                "calibrated_oof_probability": calibrated_probabilities,
            }
        ).to_csv(probability_path, index=False)
        calibration_rows.extend(
            [
                {"model": name, "probability_type": "raw", **raw_calibration},
                {"model": name, "probability_type": "calibrated_oof", **oof_calibration},
            ]
        )
        selections["models"][name] = {
            "threshold": float(selected["threshold"]),
            "validation_f1": float(selected["f1"]),
            "validation_precision": float(selected["precision"]),
            "validation_recall": float(selected["recall"]),
            "validation_raw_brier": raw_calibration["brier_score"],
            "validation_raw_ece": raw_calibration["ece"],
            "validation_oof_calibrated_brier": oof_calibration["brier_score"],
            "validation_oof_calibrated_ece": oof_calibration["ece"],
            "calibrator": calibrator_metadata,
            "validation_oof_probabilities_sha256": file_sha256(probability_path),
            "validation_oof_probabilities_path": str(probability_path.resolve().relative_to(BASE_DIR.resolve())),
        }
        logger.info(f"Frozen validation threshold for {name}: {selected['threshold']:.2f}")

    pd.DataFrame(calibration_rows).to_csv(output_dir / "validation_calibration_metrics.csv", index=False)

    with (output_dir / "thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump(selections, handle, indent=2)

    logger.info("Threshold selection complete. Test data was not loaded.")


if __name__ == "__main__":
    main()
