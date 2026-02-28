from typing import Any, Dict, List

import pandas as pd
import shap
import torch
from transformers import Pipeline, pipeline


class PhishingExplainer:
    def __init__(self, model_path: str) -> None:
        self.device = 0 if torch.cuda.is_available() else (-1 if not torch.backends.mps.is_available() else "mps")
        self.pipe: Pipeline = pipeline(
            "text-classification", model=model_path, tokenizer=model_path, device=self.device, top_k=None
        )

        self.explainer = shap.Explainer(self.pipe)

    def get_explanation(self, text: str) -> Any:
        shap_values = self.explainer([text])
        return shap_values

    def get_top_features(self, text: str, n: int = 5) -> List[Dict[str, Any]]:
        shap_values = self.get_explanation(text)
        tokens = shap_values.data[0]
        values = shap_values.values[0][:, 1]

        features = []

        for token, val in zip(tokens, values):
            features.append({"token": token, "impact": float(val)})

        return sorted(features, key=lambda x: abs(x["impact"]), reverse=True)[:n]

    def get_detailed_report(self, text: str) -> pd.DataFrame:
        shap_values = self.get_explanation(text)
        tokens = shap_values.data[0]
        scores = shap_values.values[0][:, 1]

        df = pd.DataFrame({"Token": tokens, "SHAP_Value": scores})

        df["Influence"] = df["SHAP_Value"].apply(
            lambda x: "PHISH" if x > 0.05 else ("LEGIT" if x < -0.05 else "NEUTRAL")
        )

        return df.sort_values(by="SHAP_Value", ascending=False)
