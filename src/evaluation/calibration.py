"""Validation-only probability calibration and calibration diagnostics."""

import hashlib
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

from src.config import BASE_DIR, LABEL_COL, SAVED_MODELS_DIR, SPLIT_DATA_DIR
from src.config import SPLIT_RANDOM_STATE as RANDOM_STATE
from src.config import TEMPLATE_GROUP_COL

EPSILON = 1e-6
CALIBRATION_METHOD = "sigmoid_logit"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a calibration input artefact."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_project_path(value: str) -> Path:
    """Resolve a portable project-relative artefact path."""
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def _logit_features(probabilities: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


class SigmoidProbabilityCalibrator:
    """Platt-style sigmoid calibration fitted to model log-odds."""

    def __init__(self) -> None:
        self.model = LogisticRegression(random_state=RANDOM_STATE)

    def fit(
        self,
        probabilities: NDArray[np.floating[Any]],
        labels: NDArray[np.int_],
    ) -> "SigmoidProbabilityCalibrator":
        if np.unique(labels).size != 2:
            raise ValueError("Calibration data must contain both labels")

        self.model.fit(_logit_features(probabilities), labels)
        return self

    def predict(self, probabilities: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        return np.asarray(self.model.predict_proba(_logit_features(probabilities))[:, 1], dtype=np.float64)


def cross_fitted_calibration(
    probabilities: NDArray[np.floating[Any]],
    labels: NDArray[np.int_],
    groups: NDArray[np.str_],
    n_splits: int = 5,
) -> Tuple[NDArray[np.float64], SigmoidProbabilityCalibrator, NDArray[np.int_]]:
    """Generate group-disjoint OOF calibrated probabilities on validation."""
    raw = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(labels, dtype=int)
    group_values = np.asarray(groups, dtype=str)

    if not (len(raw) == len(y_true) == len(group_values)):
        raise ValueError("Probabilities, labels, and groups must have equal lengths")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    calibrated = np.full(len(raw), np.nan, dtype=np.float64)
    fold_ids = np.full(len(raw), -1, dtype=int)

    for fold_idx, (train_idx, held_out_idx) in enumerate(
        splitter.split(np.zeros(len(raw)), y_true, groups=group_values)
    ):
        calibrator = SigmoidProbabilityCalibrator().fit(raw[train_idx], y_true[train_idx])
        calibrated[held_out_idx] = calibrator.predict(raw[held_out_idx])
        fold_ids[held_out_idx] = fold_idx

    if np.isnan(calibrated).any() or (fold_ids < 0).any():
        raise AssertionError("Every validation record must receive exactly one OOF calibrated probability")

    assignment = pd.DataFrame({"group": group_values, "fold": fold_ids})

    if assignment.groupby("group")["fold"].nunique().max() != 1:
        raise AssertionError("A template group crosses calibration folds")

    final_calibrator = SigmoidProbabilityCalibrator().fit(raw, y_true)
    return calibrated, final_calibrator, fold_ids


def reliability_table(
    labels: NDArray[np.int_],
    probabilities: NDArray[np.floating[Any]],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return fixed-width reliability bins, including empty-bin metadata."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    y_true = np.asarray(labels, dtype=int)
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    bin_ids = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    rows: List[Dict[str, Any]] = []

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        count = int(mask.sum())
        mean_probability = float(probs[mask].mean()) if count else float("nan")
        observed_frequency = float(y_true[mask].mean()) if count else float("nan")
        rows.append(
            {
                "bin": bin_idx,
                "lower": bin_idx / n_bins,
                "upper": (bin_idx + 1) / n_bins,
                "count": count,
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
                "absolute_gap": abs(mean_probability - observed_frequency) if count else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def expected_calibration_error(reliability: pd.DataFrame) -> float:
    """Calculate standard count-weighted expected calibration error."""
    total = int(reliability["count"].sum())

    if total == 0:
        raise ValueError("Cannot calculate ECE for an empty dataset")

    populated = reliability[reliability["count"] > 0]
    return float(((populated["count"] / total) * populated["absolute_gap"]).sum())


def calibration_metrics(
    labels: NDArray[np.int_],
    probabilities: NDArray[np.floating[Any]],
    n_bins: int = 10,
) -> Dict[str, float]:
    """Calculate Brier score and fixed-bin ECE."""
    table = reliability_table(labels, probabilities, n_bins=n_bins)
    return {
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(table),
    }


def plot_reliability_diagram(
    labels: NDArray[np.int_],
    raw_probabilities: NDArray[np.floating[Any]],
    calibrated_probabilities: NDArray[np.floating[Any]],
    model_name: str,
    output_path: Path,
    n_bins: int = 10,
    split_label: str = "validation OOF",
) -> None:
    """Save a dependency-free SVG reliability diagram."""
    width = height = 700
    margin = 80
    plot_size = width - 2 * margin

    def point(mean_probability: float, observed_frequency: float) -> str:
        x_coord = margin + mean_probability * plot_size
        y_coord = height - margin - observed_frequency * plot_size
        return f"{x_coord:.1f},{y_coord:.1f}"

    series: List[str] = []

    for series_name, probabilities, colour in (
        ("Raw", raw_probabilities, "#d95f02"),
        ("Calibrated", calibrated_probabilities, "#1b9e77"),
    ):
        table = reliability_table(labels, probabilities, n_bins=n_bins)
        populated = table[table["count"] > 0]
        points = " ".join(
            point(float(row["mean_probability"]), float(row["observed_frequency"])) for _, row in populated.iterrows()
        )
        circles = "".join(
            f'<circle cx="{coordinates.split(",")[0]}" cy="{coordinates.split(",")[1]}" r="5" fill="{colour}"/>'
            for coordinates in points.split()
        )
        series.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="3"/>{circles}'
            f'<text x="{width - 220}" y="{45 + 24 * len(series)}" fill="{colour}">{series_name}</text>'
        )

    title = html.escape(f"Reliability — {model_name} ({split_label})")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{title}</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{margin}" stroke="#777" stroke-dasharray="8 6"/>
<text x="{width / 2}" y="{height - 24}" text-anchor="middle">Mean predicted P(phishing)</text>
<text x="22" y="{height / 2}" text-anchor="middle" transform="rotate(-90 22 {height / 2})">Observed phishing frequency</text>
<text x="{margin}" y="{height - margin + 24}" text-anchor="middle">0</text>
<text x="{width - margin}" y="{height - margin + 24}" text-anchor="middle">1</text>
<text x="{margin - 18}" y="{height - margin + 5}" text-anchor="middle">0</text>
<text x="{margin - 18}" y="{margin + 5}" text-anchor="middle">1</text>
{''.join(series)}
</svg>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def save_calibrator(
    calibrator: SigmoidProbabilityCalibrator,
    path: Path,
    model_name: str,
    n_splits: int,
) -> Dict[str, Any]:
    """Persist a final calibrator with validation-only provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, path)
    validation_path = SPLIT_DATA_DIR / "val.csv"
    training_manifest_path = SAVED_MODELS_DIR / model_name / "training_manifest.json"

    if not training_manifest_path.exists():
        raise FileNotFoundError(f"Missing training provenance for calibration: {training_manifest_path}")

    metadata = {
        "model": model_name,
        "method": CALIBRATION_METHOD,
        "fit_split": "validation",
        "selection_predictions": "group_disjoint_out_of_fold",
        "folds": n_splits,
        "validation_sha256": file_sha256(validation_path),
        "training_manifest_sha256": file_sha256(training_manifest_path),
        "calibrator_sha256": file_sha256(path),
        "calibrator_path": str(path.resolve().relative_to(BASE_DIR.resolve())),
    }

    with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata


def load_calibrator(metadata: Dict[str, Any]) -> SigmoidProbabilityCalibrator:
    """Load a calibrator only when it matches the current validation split."""
    if metadata.get("fit_split") != "validation":
        raise ValueError("Calibrator was not fitted exclusively on validation data")

    current_hash = file_sha256(SPLIT_DATA_DIR / "val.csv")

    if metadata.get("validation_sha256") != current_hash:
        raise ValueError("Calibrator is stale: validation split hash changed")

    model_name = str(metadata["model"])
    training_manifest_path = SAVED_MODELS_DIR / model_name / "training_manifest.json"

    if not training_manifest_path.exists():
        raise FileNotFoundError(f"Missing current training manifest for calibrated model: {model_name}")

    if metadata.get("training_manifest_sha256") != file_sha256(training_manifest_path):
        raise ValueError("Calibrator is stale: model training manifest changed")

    path = resolve_project_path(str(metadata["calibrator_path"]))

    if metadata.get("calibrator_sha256") != file_sha256(path):
        raise ValueError("Calibrator content hash changed")

    calibrator = joblib.load(path)

    if not isinstance(calibrator, SigmoidProbabilityCalibrator):
        raise TypeError(f"Unexpected calibrator type in {path}")

    return calibrator
