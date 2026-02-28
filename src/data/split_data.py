from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    LABEL_COL,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SPLIT_DATA_DIR,
    TEXT_COL,
)
from src.data.augment_data import PhishingAugmenter
from src.features.extractor import PhishingFeatureExtractor
from src.utils.logger import get_logger

# Setup logging:
logger = get_logger(__name__)


# Helper functions:
def get_latest_split() -> pd.DataFrame:
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


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate dataset structure and label integrity.

    Ensures that required columns are present and that the label column
    does not contain missing values.

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

    logger.info(f"Dataset validation passed. Size: {len(df)}")
    logger.info(f"Class distribution:\n{df[LABEL_COL].value_counts(normalize=True)}\n")


def split_dataset(
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
        ValueError: If the provided split proportions do not sum to 1.0.
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError(f"Split proportions must sum to 1.0. Got: {train_size + val_size + test_size}")

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


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save dataset splits to CSV files.

    Args:
        train_df: Training dataset split.
        val_df: Validation dataset split.
        test_df: Test dataset split.
    """
    SPLIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(SPLIT_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_DATA_DIR / "test.csv", index=False)

    logger.info(f"Splits saved to {SPLIT_DATA_DIR}")
    logger.info(f"Train: \t{len(train_df)}")
    logger.info(f"Val: \t{len(val_df)}")
    logger.info(f"Test: \t{len(test_df)}")


# Main:
def main() -> None:
    logger.info("Starting dataset split and late-stage feature extraction...")

    df = get_latest_split()
    validate_dataframe(df)

    # 1. ...
    train_df, val_df, test_df = split_dataset(df)

    # 2. ...
    augmenter = PhishingAugmenter(aug_prob=0.1)
    extractor = PhishingFeatureExtractor()

    def process_row(text: str, is_phishing: bool, augment: bool) -> str:
        # A. ...
        parts = text.split("[CONTENT] ", 1)
        meta = parts[0]
        content = parts[1] if len(parts) > 1 else ""

        # B. ...
        if augment and is_phishing:
            content = augmenter.augment(content)

        # C. ...
        features = extractor.get_all_features(content)
        feat_tags = (
            f"[FEAT: URG={features['urgency_score']} "
            f"THR={features['threat_score']} "
            f"VER={features['verification_request']} "
            f"TLD={features['has_suspicious_tld']} "
            f"HOMO={features['has_homograph_attack']}]"
        )

        # D. ...
        return f"{feat_tags}\n{meta}[CONTENT] {content}"

    logger.info("Applying augmentation and extracting features for Train...")
    train_df[TEXT_COL] = train_df.apply(lambda row: process_row(row[TEXT_COL], row[LABEL_COL], augment=True), axis=1)

    logger.info("Extracting features for Val and Test (no augmentation)...")
    for d in [val_df, test_df]:
        d[TEXT_COL] = d.apply(lambda row: process_row(row[TEXT_COL], row[LABEL_COL], augment=False), axis=1)

    save_splits(train_df, val_df, test_df)


# Entry point:
if __name__ == "__main__":
    main()
