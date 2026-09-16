"""Final per-model evaluation using validation-frozen thresholds."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.config import (
    BASE_DIR,
    LABEL_COL,
    RANDOM_STATE,
    RESULTS_DIR,
    SOURCE_COL,
    SPLIT_DATA_DIR,
    TEMPLATE_GROUP_COL,
    TEXT_COL,
)
from src.data.load_data import load_split, prepare_xy
from src.evaluation.calibration import calibration_metrics, load_calibrator, plot_reliability_diagram
from src.evaluation.inference import discover_models, predict_model
from src.evaluation.uncertainty import (
    grouped_bootstrap_confidence_intervals,
    template_group_binomial_confidence_intervals,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_final_metrics(
    y_true: NDArray[np.int_], probabilities: NDArray[np.floating[Any]], threshold: float
) -> Dict[str, float]:
    """Calculate the single frozen set of final classification metrics."""
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    roc_auc = float(roc_auc_score(y_true, probabilities)) if np.unique(y_true).size == 2 else float("nan")
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "roc_auc": roc_auc,
        "errors": int((predicted != y_true).sum()),
        "false_positives": int(((predicted == 1) & (y_true == 0)).sum()),
        "false_negatives": int(((predicted == 0) & (y_true == 1)).sum()),
        "tpr": float(tp / (tp + fn)) if tp + fn else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
        "tnr": float(tn / (tn + fp)) if tn + fp else 0.0,
    }


def _load_frozen_configuration() -> Dict[str, Any]:
    path = RESULTS_DIR / "threshold_selection" / "thresholds.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing validation threshold selection: {path}. Run the 'threshold' step before final evaluation."
        )

    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, dict):
        raise TypeError(f"Invalid frozen configuration in {path}")

    payload: Dict[str, Any] = loaded

    if payload.get("selection_split") != "validation":
        raise ValueError("Threshold configuration was not selected on validation data")

    calibration = payload.get("calibration", {})

    if calibration.get("fit_split") != "validation":
        raise ValueError("Probability calibration was not fitted on validation data")

    if payload.get("selection_predictions") != "group_disjoint_calibration_oof":
        raise ValueError("Thresholds were not selected from group-disjoint calibrated OOF predictions")

    return payload


def _slice_metrics(
    df: pd.DataFrame,
    probabilities: NDArray[np.floating[Any]],
    threshold: float,
    model_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for column in (SOURCE_COL, "Type"):
        if column not in df.columns:
            continue

        for value, indices in df.groupby(column, sort=True).groups.items():
            positions = np.asarray(list(indices), dtype=int)
            labels = df.iloc[positions][LABEL_COL].astype(int).to_numpy()
            probs = probabilities[positions]
            metrics = calculate_final_metrics(labels, probs, threshold)
            rows.append({"model": model_name, "slice": column, "value": value, "n": len(positions), **metrics})

    return rows


def main(model_uri: str = "", models_dir: str = "", threshold: Optional[float] = None) -> None:
    """Evaluate registered models once; all choices must already be frozen."""
    if model_uri or threshold is not None:
        raise ValueError(
            "Final evaluation does not accept ad-hoc model URIs or threshold overrides. "
            "Register the model and select its threshold on validation data first."
        )

    from src.evaluation.ensemble import average_probabilities
    from src.utils.artifacts import atomic_json, bind_run, file_sha256

    bind_run()
    frozen = _load_frozen_configuration()
    frozen_models = frozen.get("models", {})
    root = Path(models_dir) if models_dir else None
    models = discover_models(root)

    if not models:
        raise RuntimeError("No complete registered models found")

    calibrators = {str(m["name"]): load_calibrator(dict(frozen_models[str(m["name"])]["calibrator"])) for m in models}
    ensemble_path = RESULTS_DIR / "ensemble_selection" / "ensemble_config.json"
    ensemble = json.loads(ensemble_path.read_text())

    if ensemble["threshold_configuration_sha256"] != file_sha256(
        RESULTS_DIR / "threshold_selection" / "thresholds.json"
    ):
        raise ValueError("Ensemble configuration is stale")

    if not set(ensemble["models"]) <= set(calibrators):
        raise ValueError("Missing ensemble members")

    # Test data is loaded only after the selection artefacts have been validated:
    test_df = load_split("test").reset_index(drop=True)
    x_test, y_test = prepare_xy(test_df)
    aggregate_rows: List[Dict[str, Any]] = []
    slice_rows: List[Dict[str, Any]] = []
    confidence_rows: List[Dict[str, Any]] = []
    group_binomial_rows: List[Dict[str, Any]] = []
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    with (BASE_DIR / "params.yaml").open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle).get("research_protocol", {})

    uncertainty = protocol.get("final_test_uncertainty", {})

    if uncertainty.get("unit") != TEMPLATE_GROUP_COL:
        raise ValueError(f"Final test uncertainty must resample {TEMPLATE_GROUP_COL}")

    bootstrap_resamples = int(uncertainty.get("resamples", 5000))
    confidence_level = float(uncertainty.get("confidence_level", 0.95))
    ece_bins = int(frozen.get("calibration", {}).get("ece_bins", 10))
    prediction_rows = []
    calibrated_by_name: Dict[str, NDArray[np.float64]] = {}
    raw_by_name: Dict[str, NDArray[np.float64]] = {}

    for model in [*models, {"name": "ensemble", "type": "ensemble"}]:
        name = str(model["name"])

        if name != "ensemble" and name not in frozen_models:
            raise ValueError(f"No frozen validation threshold for model: {name}")

        if name == "ensemble":
            threshold_value = float(ensemble["threshold"])
            raw_probabilities = average_probabilities(raw_by_name, ensemble["models"])
            probabilities = average_probabilities(calibrated_by_name, ensemble["models"])

        else:
            model_config = frozen_models[name]
            threshold_value = float(model_config["threshold"])
            raw_probabilities = predict_model(model, x_test)
            probabilities = calibrators[name].predict(raw_probabilities)
            raw_by_name[name] = raw_probabilities
            calibrated_by_name[name] = probabilities

        frame = test_df[["Record_ID", TEMPLATE_GROUP_COL, SOURCE_COL, "Type", LABEL_COL, TEXT_COL]].copy()
        frame["model"] = name
        frame["raw_probability"] = raw_probabilities
        frame["probability"] = probabilities
        frame["threshold"] = threshold_value
        frame["prediction"] = (probabilities >= threshold_value).astype(int)
        prediction_rows.append(frame)
        metrics = calculate_final_metrics(y_test.to_numpy(), probabilities, threshold_value)
        metrics.update(calibration_metrics(y_test.to_numpy(), probabilities, n_bins=ece_bins))

        intervals = grouped_bootstrap_confidence_intervals(
            y_test.to_numpy(dtype=int),
            probabilities,
            test_df[TEMPLATE_GROUP_COL].astype(str).to_numpy(),
            threshold=threshold_value,
            n_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            random_state=RANDOM_STATE,
        )

        for metric_name, interval in intervals.items():
            metrics[f"{metric_name}_ci_lower"] = interval["ci_lower"]
            metrics[f"{metric_name}_ci_upper"] = interval["ci_upper"]
            confidence_rows.append(
                {
                    "model": name,
                    "split": "final_test",
                    "method": "Template_Group_cluster_bootstrap_percentile",
                    "metric": metric_name,
                    **interval,
                }
            )

        group_intervals = template_group_binomial_confidence_intervals(
            y_test.to_numpy(dtype=int),
            probabilities,
            test_df[TEMPLATE_GROUP_COL].astype(str).to_numpy(),
            threshold=threshold_value,
            confidence_level=confidence_level,
        )

        for metric_name, interval in group_intervals.items():
            metrics[metric_name] = interval["estimate"]
            metrics[f"{metric_name}_ci_lower"] = interval["ci_lower"]
            metrics[f"{metric_name}_ci_upper"] = interval["ci_upper"]
            group_binomial_rows.append(
                {
                    "model": name,
                    "split": "final_test",
                    "method": "Template_Group_Clopper_Pearson_exact",
                    "metric": metric_name,
                    **interval,
                }
            )

        aggregate_rows.append({"model": name, "split": "final_test", **metrics})
        slice_rows.extend(_slice_metrics(test_df, probabilities, threshold_value, name))
        plot_reliability_diagram(
            y_test.to_numpy(),
            raw_probabilities,
            probabilities,
            name,
            results_dir / f"final_test_reliability_{name}.svg",
            n_bins=ece_bins,
            split_label="final test (descriptive)",
        )

        logger.info(f"Final test {name}: F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")

    pd.concat(prediction_rows, ignore_index=True).to_csv(results_dir / "final_test_predictions.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(results_dir / "final_test_metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(results_dir / "final_test_slice_metrics.csv", index=False)
    pd.DataFrame(confidence_rows).to_csv(results_dir / "final_test_group_bootstrap_ci.csv", index=False)
    pd.DataFrame(group_binomial_rows).to_csv(results_dir / "final_test_group_binomial_ci.csv", index=False)
    atomic_json(
        results_dir / "final_test_manifest.json",
        {
            "test_sha256": file_sha256(SPLIT_DATA_DIR / "test.csv"),
            "selection_sha256": file_sha256(results_dir / "threshold_selection" / "thresholds.json"),
            "ensemble_sha256": file_sha256(ensemble_path),
            "predictions_sha256": file_sha256(results_dir / "final_test_predictions.csv"),
            "test_status": "exploratory_reanalysis_previously_inspected_corpus",
        },
    )


def load_final_predictions() -> pd.DataFrame:
    """Verify the saved final evaluation before plotting or explaining it."""
    from src.models.provenance import current_split_manifest
    from src.utils.artifacts import file_sha256, verify_run

    if not (RESULTS_DIR / "final_test_manifest.json").exists():
        raise FileNotFoundError(
            "No completed v2 evaluation. Run preprocessing, training, threshold, ensemble and evaluate stages first."
        )
    verify_run()
    current_split_manifest(verify_splits=("train", "val", "test"))
    manifest = json.loads((RESULTS_DIR / "final_test_manifest.json").read_text())

    for key, path in {
        "test_sha256": SPLIT_DATA_DIR / "test.csv",
        "selection_sha256": RESULTS_DIR / "threshold_selection" / "thresholds.json",
        "ensemble_sha256": RESULTS_DIR / "ensemble_selection" / "ensemble_config.json",
        "predictions_sha256": RESULTS_DIR / "final_test_predictions.csv",
    }.items():
        if manifest[key] != file_sha256(path):
            raise ValueError(f"Final evaluation input/output changed: {path}")

    frozen = _load_frozen_configuration()

    for model in discover_models():
        load_calibrator(frozen["models"][str(model["name"])]["calibrator"])

    frame = pd.read_csv(RESULTS_DIR / "final_test_predictions.csv")
    expected_ids = set(load_split("test")["Record_ID"])

    if set(frame.model) != {*frozen["models"], "ensemble"}:
        raise ValueError("Final evaluation model scope is incomplete")

    for _, rows in frame.groupby("model"):
        if set(rows.Record_ID) != expected_ids or rows.Record_ID.duplicated().any():
            raise ValueError("Final prediction coverage changed")

    return frame


if __name__ == "__main__":
    main()
