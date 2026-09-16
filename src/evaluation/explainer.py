"""SHAP explanations of the same calibrated probability used for decisions."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from src.config import RESULTS_DIR
from src.evaluation.calibration import load_calibrator
from src.evaluation.inference import TransformerPredictor
from src.models.provenance import artefact_matches_current_split


class PhishingExplainer:
    def __init__(self, model_path: str) -> None:
        path = Path(model_path)

        if not artefact_matches_current_split(path):
            raise ValueError("XAI requires a complete model from the current run")

        frozen = json.loads((RESULTS_DIR / "threshold_selection" / "thresholds.json").read_text())
        config = frozen["models"][path.name]
        self.calibrator = load_calibrator(config["calibrator"])
        self.threshold = float(config["threshold"])
        self.session = TransformerPredictor(path)
        self.explainer = shap.Explainer(
            self.predict, shap.maskers.Text(self.session.tokenizer), output_names=["LEGIT", "PHISH"], seed=42
        )

    def predict(self, texts: Any) -> np.ndarray:
        raw = self.session.predict([str(text) for text in texts])
        probabilities = self.calibrator.predict(raw)
        return np.column_stack((1.0 - probabilities, probabilities))

    def decision(self, text: str) -> dict[str, Any]:
        probability = float(self.predict([text])[0, 1])
        return {
            "probability": probability,
            "threshold": self.threshold,
            "prediction": int(probability >= self.threshold),
        }

    def get_explanation(self, text: str) -> Any:
        return self.explainer([text])

    def get_top_features(self, text: str, n: int = 5) -> list[dict[str, Any]]:
        report = self.get_detailed_report(text)
        return [
            {"token": row.Token, "impact": float(row.SHAP_Value)}
            for row in report.reindex(report.SHAP_Value.abs().sort_values(ascending=False).index).head(n).itertuples()
        ]

    def get_detailed_report(self, text: str) -> pd.DataFrame:
        values = self.get_explanation(text)
        result = pd.DataFrame({"Token": values.data[0], "SHAP_Value": values.values[0][:, 1]})
        result["Influence"] = np.where(
            result.SHAP_Value > 0, "PHISH", np.where(result.SHAP_Value < 0, "LEGIT", "NEUTRAL")
        )
        return result.sort_values("SHAP_Value", ascending=False)

    def close(self) -> None:
        self.session.close()
