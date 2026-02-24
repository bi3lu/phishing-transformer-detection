from typing import Tuple

import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from src.config import (
    LABEL_COL,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SPLIT_DATA_DIR,
    TEXT_COL,
)
from src.utils.logger import get_logger

# Setup logging:
logger = get_logger(__name__)


# Helper functions:
def _load_latest_processed_file() -> pd.DataFrame:
    """Load the most recent processed dataset file.

    The function searches for files matching the pattern
    ``processed_data_*.csv`` in ``PROCESSED_DATA_DIR``, selects the
    latest one based on lexicographic sorting, and loads it into a
    pandas DataFrame.

    Returns:
        A pandas DataFrame containing the most recently processed data.

    Raises:
        FileNotFoundError: If no matching processed files are found.
    """
    files = sorted(PROCESSED_DATA_DIR.glob("processed_data_*.csv"))

    if not files:
        raise FileNotFoundError("No processed_data_*.csv files found.")

    latest_file = files[-1]

    logger.info(f"Using processed file: {latest_file}")

    return pd.read_csv(latest_file)


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate dataset structure and label integrity.

    Ensures that required columns are present and that the label column
    does not contain missing values. Logs dataset size and normalized
    class distribution for inspection.

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If required columns are missing or the label column
            contains NaN values.
    """
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets.

    The split is stratified by the label column to preserve class
    distribution across subsets. The proportions must sum to 1.0.

    Args:
        df: Full input dataset.
        train_size: Proportion of data assigned to the training set.
        val_size: Proportion of data assigned to the validation set.
        test_size: Proportion of data assigned to the test set.

    Returns:
        A tuple (train_df, val_df, test_df) containing the respective
        dataset splits.

    Raises:
        AssertionError: If the provided split proportions do not sum to 1.0.
    """
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
    """Execute dataset splitting pipeline.

    Loads the most recent processed dataset, validates its structure,
    performs a stratified split into train, validation, and test sets,
    and saves each subset as a separate CSV file in
    ``SPLIT_DATA_DIR``.

    Returns:
        None
    """
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
