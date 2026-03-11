"""Loading and preparation of phishing detection dataset splits.

Provides utilities to load pre-split datasets and extract feature/label pairs.
"""

from typing import Tuple

import pandas as pd

from src.config import LABEL_COL, SPLIT_DATA_DIR, TEXT_COL


def load_split(name: str) -> pd.DataFrame:
    """Load a dataset split from a CSV file.

    The function expects a CSV file named ``{name}.csv`` to exist inside
    ``SPLIT_DATA_DIR``. It reads the file into a pandas DataFrame.

    Args:
        name: Name of the split to load (e.g., "train", "val", "test").

    Returns:
        A pandas DataFrame containing the requested dataset split.

    Raises:
        FileNotFoundError: If the expected CSV file does not exist.
    """
    path = SPLIT_DATA_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    return pd.read_csv(path)


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Prepare feature and label vectors from a DataFrame.

    Extracts the text column as input features and converts the label
    column to integer-encoded binary targets (False → 0, True → 1).

    Args:
        df: Input DataFrame containing at least the configured text and
            label columns.

    Returns:
        A tuple (X, y), where:
            - X: pandas Series of text data (strings).
            - y: pandas Series of integer labels (0 or 1).
    """
    X = df[TEXT_COL].astype(str)
    y = df[LABEL_COL].map({False: 0, True: 1}).astype(int)

    return X, y
