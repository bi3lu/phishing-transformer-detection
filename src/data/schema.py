"""Strict validation at ingestion and whenever a split is loaded."""

import pandas as pd

from src.config import CONTENT_COL, LABEL_COL, SOURCE_COL, TEXT_COL


def validate_records(df: pd.DataFrame, require_text: bool = True) -> None:
    required = {"Type", "Title", CONTENT_COL, LABEL_COL, SOURCE_COL}
    if require_text:
        required.add(TEXT_COL)

    if required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")

    if df.empty:
        raise ValueError("Dataset is empty")

    for column in required - {"Title"}:
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing/empty values in {column}")

    if not df[LABEL_COL].isin([0, 1, False, True]).all():
        raise ValueError("Labels must be binary 0/1")

    if not df["Type"].isin(["SMS", "E-mail", "Notification"]).all():
        raise ValueError("Type must be SMS, E-mail or Notification")

    if require_text and df[TEXT_COL].str.contains(r"\[FEAT|\[SENDER\]", regex=True).any():
        raise ValueError("Legacy generator/feature tags in model input")
