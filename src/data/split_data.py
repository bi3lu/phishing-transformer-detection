"""Dataset splitting and late-stage feature extraction.

Loads processed phishing data, splits it into train/validation/test sets,
applies data augmentation to training data, extracts phishing indicators,
and saves the processed splits.
"""

import random
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
from src.data.adversarial_augment.adversarial_generator import AdversarialAugmenter
from src.data.augment.augment_data import PhishingAugmenter
from src.features.extractor import PhishingFeatureExtractor
from src.utils.logger import get_logger

# Setup logging:
logger = get_logger(__name__)

# Constants:
FEATURE_DROPOUT_PROB = 0.3


# Helper functions:
def get_latest_split() -> pd.DataFrame:
    files = sorted(PROCESSED_DATA_DIR.glob("processed_data_*.csv"))

    if not files:
        raise FileNotFoundError("No processed_data_*.csv files found.")

    latest_file = files[-1]
    logger.info(f"Using processed file: {latest_file}")

    return pd.read_csv(latest_file)


def validate_dataframe(df: pd.DataFrame) -> None:
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
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError(f"Split proportions must sum to 1.0. Got: {train_size + val_size + test_size}")

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_size), stratify=df[LABEL_COL], random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        stratify=temp_df[LABEL_COL],
        random_state=RANDOM_STATE,
    )

    return (train_df, val_df, test_df)


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
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

    # 1. Split dataset:
    train_df, val_df, test_df = split_dataset(df)

    # 2. Initialize tools:
    augmenter = PhishingAugmenter(aug_prob=0.1)
    extractor = PhishingFeatureExtractor()

    def _process_row(text: str, is_phishing: bool, augment: bool, is_train: bool = True) -> str:
        parts = text.split("[CONTENT] ", 1)
        meta = parts[0]
        content = parts[1] if len(parts) > 1 else ""

        # Apply adversarial augmentation to all phishing samples:
        if augment and is_phishing:
            if random.random() < 0.2:
                content = AdversarialAugmenter.generate_hard_phish(content)

            else:
                content = augmenter.augment(content)

        # Extract features (always calculated to ensure extractor consistency):
        features = extractor.get_all_features(content)

        # Build enriched feature tags (Behavioral + Technical):
        feat_tags = (
            f"[FEAT: URG={features['urgency_score']} "
            f"THR={features['threat_score']} "
            f"VER={features['verif_score']} "
            f"ACT={features['action_score']} "
            f"FIN={features['fin_score']} "
            f"EMO={features['emo_score']} "
            f"TLD={features['has_suspicious_tld']} "
            f"HOMO={features['has_homograph_attack']}]"
        )

        # Logic for Feature Dropout:
        use_tags = True

        if is_train and random.random() < FEATURE_DROPOUT_PROB:
            use_tags = False

        if use_tags:
            return f"{feat_tags}\n{meta}[CONTENT] {content}"

        else:
            return f"{meta}[CONTENT] {content}"

    logger.info("Processing Train set (with Augmentation and 30% Feature Dropout)...")
    train_df[TEXT_COL] = train_df.apply(
        lambda row: _process_row(row[TEXT_COL], row[LABEL_COL], augment=True, is_train=True), axis=1
    )

    logger.info("Processing Val and Test sets (Full features, no Dropout, no Augmentation)...")

    for d in [val_df, test_df]:
        d[TEXT_COL] = d.apply(
            lambda row: _process_row(row[TEXT_COL], row[LABEL_COL], augment=False, is_train=False), axis=1
        )

    save_splits(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
