import logging
from pathlib import Path
from typing import Tuple

import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from src.config import (LABEL_COL, PROCESSED_DATA_DIR, RANDOM_STATE,
                        SPLIT_DATA_DIR, TEXT_COL)
from src.utils.logger import get_logger

# Setup logging:
logger = get_logger(__name__)


# Helper functions:
def _load_latest_processed_file() -> pd.DataFrame:  # TODO: Add docstring
    files = sorted(PROCESSED_DATA_DIR.glob("processed_data_*.csv"))

    if not files:
        raise FileNotFoundError("No processed_data_*.csv files found.")

    latest_file = files[-1]

    logger.info(f"Using processed file: {latest_file}")

    return pd.read_csv(latest_file)


def _validate_dataframe(df: pd.DataFrame) -> None:  # TODO: Add docstring
    missing_cols = {LABEL_COL, TEXT_COL} - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}.")

    if df[LABEL_COL].isnull().any():
        raise ValueError("Label column contains NaN values.")

    logger.info(f"Dataset size: {len(df)}")
    logger.info(f"Class distribution:\n{df[LABEL_COL].value_counts(normalize=True)}\n")


def _split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # TODO: Add docstring
    assert (
        abs(train_size + val_size + test_size - 1.0) < 1e-6
    )  # NOTE: Checks if train, validate and test sizes are correct

    train_df, temp_df = train_test_split(
        df, test_size=(1 - test_size), stratify=df[LABEL_COL], random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        stratify=temp_df[LABEL_COL],
        random_state=RANDOM_STATE,
    )

    return (train_df, val_df, test_df)


# Main:
def main() -> None:
    logger.info("Starting dataset split...")

    # Load and check dataset:
    df = _load_latest_processed_file()
    _validate_dataframe(df)

    # Split to train, validate and test datasets
    train_df, val_df, test_df = _split_dataset(df)

    SPLIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(SPLIT_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_DATA_DIR / "test.csv", index=False)

    logger.info("Split completed successfully:")
    logger.info(f"Train: \t{len(train_df)}")
    logger.info(f"Val: \t{len(val_df)}")
    logger.info(f"Test: \t{len(test_df)}")


# Entry point:
if __name__ == "__main__":
    main()
