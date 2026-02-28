"""Explainability tools for phishing detection model predictions.

Uses SHAP (SHapley Additive exPlanations) to provide interpretable
feature importance for individual predictions.
"""

from typing import Any, Dict, List

import pandas as pd
import shap
import torch
from transformers import Pipeline, pipeline


class PhishingExplainer:
    """Explains phishing model predictions using SHAP values.

    Loads a transformer model and provides token-level importance
    scores for understanding model predictions.
    """

    def __init__(self, model_path: str) -> None:
        """Initialize the explainer with a model.

        Args:
            model_path: Path to the transformer model directory.
        """
        self.device = 0 if torch.cuda.is_available() else (-1 if not torch.backends.mps.is_available() else "mps")
        self.pipe: Pipeline = pipeline(
            "text-classification", model=model_path, tokenizer=model_path, device=self.device, top_k=None
        )

        self.explainer = shap.Explainer(self.pipe)

    def get_explanation(self, text: str) -> Any:
        """Get SHAP explanation for a text sample.

        Args:
            text: Text to explain.

        Returns:
            SHAP Explainer output with feature importance values.
        """
        shap_values = self.explainer([text])
        return shap_values

    def get_top_features(self, text: str, n: int = 5) -> List[Dict[str, Any]]:
        """Get the top N most influential tokens for a prediction.

        Args:
            text: Text to explain.
            n: Number of top features to return. Defaults to 5.

        Returns:
            List of dictionaries with token and impact score.
        """
        shap_values = self.get_explanation(text)
        tokens = shap_values.data[0]
        values = shap_values.values[0][:, 1]

        features = []

        for token, val in zip(tokens, values):
            features.append({"token": token, "impact": float(val)})

        return sorted(features, key=lambda x: abs(x["impact"]), reverse=True)[:n]

    def get_detailed_report(self, text: str) -> pd.DataFrame:
        """Generate a detailed DataFrame of all token contributions.

        Args:
            text: Text to analyze.

        Returns:
            DataFrame with tokens, SHAP values, and influence labels.
        """
        shap_values = self.get_explanation(text)
        tokens = shap_values.data[0]
        scores = shap_values.values[0][:, 1]

        df = pd.DataFrame({"Token": tokens, "SHAP_Value": scores})

        df["Influence"] = df["SHAP_Value"].apply(
            lambda x: "PHISH" if x > 0.05 else ("LEGIT" if x < -0.05 else "NEUTRAL")
        )

        return df.sort_values(by="SHAP_Value", ascending=False)
