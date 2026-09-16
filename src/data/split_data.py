"""Leakage-safe train/validation/test splitting.

Records are assigned by normalized template group, never by individual row.
Consequently exact duplicates and messages that differ only in URLs, amounts,
dates, or identifiers cannot cross split boundaries.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    BASE_DIR,
    CONTENT_COL,
    LABEL_COL,
    PROCESSED_DATA_DIR,
    PROTOCOL_VERSION,
    SOURCE_COL,
    SPLIT_DATA_DIR,
)
from src.config import SPLIT_RANDOM_STATE as RANDOM_STATE
from src.config import (
    TEMPLATE_GROUP_COL,
    TEXT_COL,
)
from src.data.hygiene import add_template_groups, assert_group_label_consistency
from src.data.schema import validate_records
from src.utils.artifacts import atomic_json, implementation_identity
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_GROUP_FOLDS = 10


def get_latest_processed() -> Tuple[pd.DataFrame, Path]:
    """Load the newest canonical processed dataset."""
    files = sorted(PROCESSED_DATA_DIR.glob("processed_data_*.csv"))

    if not files:
        raise FileNotFoundError("No processed_data_*.csv files found.")

    latest_file = files[-1]
    logger.info(f"Using processed file: {latest_file}")
    return pd.read_csv(latest_file), latest_file


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate schema, labels, template groups, and exact uniqueness."""
    validate_records(df)
    assert_group_label_consistency(df)


def _build_strata(df: pd.DataFrame, n_splits: int) -> pd.Series:
    """Build label strata; source balance is audited rather than forced.

    Several generators contain too few positive or negative groups to appear
    in every fold. Treating generator/label pairs as classes would therefore
    produce statistically invalid sparse strata. Generator generalisation is
    reported separately through source slices and dedicated holdout studies.
    """
    del n_splits
    return "label_" + df[LABEL_COL].astype(int).astype(str)


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    n_splits: int = DEFAULT_GROUP_FOLDS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic stratified splits with disjoint template groups."""
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError(f"Split proportions must sum to 1.0. Got: {train_size + val_size + test_size}")

    if TEMPLATE_GROUP_COL not in df.columns:
        df = add_template_groups(df)

    assert_group_label_consistency(df)

    if abs(val_size - test_size) > 1e-6:
        raise ValueError("The two-stage group splitter currently requires equal validation and test proportions")

    train_folds = round(train_size * n_splits)

    if train_folds < 1 or train_folds >= n_splits:
        raise ValueError(f"n_splits={n_splits} is too small for requested proportions")

    strata = _build_strata(df, n_splits)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_id = pd.Series(index=df.index, dtype="int64")
    dummy_x = pd.Series(0, index=df.index)

    for current_fold, (_, held_out_idx) in enumerate(splitter.split(dummy_x, strata, groups=df[TEMPLATE_GROUP_COL])):
        fold_id.iloc[held_out_idx] = current_fold

    train_df = df.loc[fold_id < train_folds].copy()
    temporary_df = df.loc[fold_id >= train_folds].copy().reset_index(drop=True)

    # Split the held-out 30% into two balanced, group-disjoint halves.
    temp_strata = _build_strata(temporary_df, n_splits=2)
    temp_splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE + 1)
    temp_x = pd.Series(0, index=temporary_df.index)
    val_idx, test_idx = next(temp_splitter.split(temp_x, temp_strata, groups=temporary_df[TEMPLATE_GROUP_COL]))
    val_df = temporary_df.iloc[val_idx].copy()
    test_df = temporary_df.iloc[test_idx].copy()

    for split in (train_df, val_df, test_df):
        split.sort_values([TEMPLATE_GROUP_COL, SOURCE_COL, CONTENT_COL], inplace=True)
        split.reset_index(drop=True, inplace=True)

    assert_disjoint_splits(train_df, val_df, test_df)
    return train_df, val_df, test_df


def assert_disjoint_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Fail fast if content or template groups cross any split boundary."""
    named = {"train": train_df, "val": val_df, "test": test_df}
    names = list(named)

    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left, right = named[left_name], named[right_name]
            group_overlap = set(left[TEMPLATE_GROUP_COL]) & set(right[TEMPLATE_GROUP_COL])
            content_overlap = set(left[CONTENT_COL]) & set(right[CONTENT_COL])

            if group_overlap or content_overlap:
                raise AssertionError(
                    f"Leakage between {left_name}/{right_name}: "
                    f"groups={len(group_overlap)}, exact_content={len(content_overlap)}"
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    source_path: Path,
) -> None:
    """Save splits plus a machine-readable provenance and leakage manifest."""
    existing = SPLIT_DATA_DIR / "split_manifest.json"

    if existing.exists():
        old = json.loads(existing.read_text())

        if old.get("source_sha256") != _sha256(source_path) or old.get("implementation") != implementation_identity():
            raise ValueError(
                "Versioned splits already exist for different data/code. Archive data/v2 before rebuilding."
            )

    SPLIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    splits: Dict[str, pd.DataFrame] = {"train": train_df, "val": val_df, "test": test_df}
    paths: Dict[str, Path] = {}

    for name, split in splits.items():
        paths[name] = SPLIT_DATA_DIR / f"{name}.csv"
        split.to_csv(paths[name], index=False)

    manifest: Dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "implementation": implementation_identity(),
        "test_status": "exploratory_reanalysis_previously_inspected_corpus",
        "method": "Two-stage StratifiedGroupKFold allocation; template groups are disjoint",
        "random_state": RANDOM_STATE,
        "group_column": TEMPLATE_GROUP_COL,
        "source_dataset": str(source_path.resolve().relative_to(BASE_DIR)),
        "source_sha256": _sha256(source_path),
        "splits": {},
    }
    split_manifest: Dict[str, object] = {}

    for name, split in splits.items():
        split_manifest[name] = {
            "rows": len(split),
            "template_groups": int(split[TEMPLATE_GROUP_COL].nunique()),
            "positive": int(split[LABEL_COL].astype(int).sum()),
            "sha256": _sha256(paths[name]),
            "source_counts": {str(k): int(v) for k, v in split[SOURCE_COL].value_counts().items()},
        }
    manifest["splits"] = split_manifest

    with (SPLIT_DATA_DIR / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    logger.info(f"Leakage-safe splits saved to {SPLIT_DATA_DIR}")

    for name, split in splits.items():
        logger.info(f"{name}: {len(split)} rows, {split[TEMPLATE_GROUP_COL].nunique()} groups")


def main() -> None:
    """Build canonical leakage-safe splits without label-dependent augmentation."""
    logger.info("Starting leakage-safe stratified group split...")
    df, source_path = get_latest_processed()

    if TEMPLATE_GROUP_COL not in df.columns:
        df = add_template_groups(df)

    validate_dataframe(df)
    train_df, val_df, test_df = split_dataset(df)
    save_splits(train_df, val_df, test_df, source_path)


if __name__ == "__main__":
    main()
