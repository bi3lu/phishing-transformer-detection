"""Lightweight, leakage-safe fold assignment utilities."""

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from src.config import LABEL_COL, SOURCE_COL
from src.config import SPLIT_RANDOM_STATE as RANDOM_STATE
from src.config import TEMPLATE_GROUP_COL, TEXT_COL

TEMPLATE_GROUP_PROTOCOL = "template_group"
SOURCE_HOLDOUT_PROTOCOL = "source_holdout"


def _source_label_strata(df: pd.DataFrame, n_splits: int) -> pd.Series:
    """Balance source/label pairs, pooling sources too small for all folds."""
    group_table = df.drop_duplicates([TEMPLATE_GROUP_COL, SOURCE_COL, LABEL_COL])
    pair_counts = group_table.groupby([SOURCE_COL, LABEL_COL]).size()
    sources = sorted(group_table[SOURCE_COL].astype(str).unique())
    labels = sorted(group_table[LABEL_COL].unique())
    rare_sources = {
        source for source in sources if any(int(pair_counts.get((source, label), 0)) < n_splits for label in labels)
    }

    remaining = sorted(
        (source for source in sources if source not in rare_sources),
        key=lambda source: int((group_table[SOURCE_COL].astype(str) == source).sum()),
    )

    while rare_sources and any(
        int(
            group_table[
                group_table[SOURCE_COL].astype(str).isin(rare_sources) & (group_table[LABEL_COL] == label)
            ].shape[0]
        )
        < n_splits
        for label in labels
    ):
        if not remaining:
            break

        rare_sources.add(remaining.pop(0))

    source_stratum = (
        df[SOURCE_COL].astype(str).where(~df[SOURCE_COL].astype(str).isin(rare_sources), "__pooled_rare_sources__")
    )

    return source_stratum + "|label=" + df[LABEL_COL].astype(int).astype(str)


def build_cv_folds(
    df: pd.DataFrame,
    n_splits: int,
    protocol: str = TEMPLATE_GROUP_PROTOCOL,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Return one held-out fold ID per row and verify group isolation."""
    if protocol == TEMPLATE_GROUP_PROTOCOL:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        strata = _source_label_strata(df, n_splits)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iterator = splitter.split(df[TEXT_COL], strata, groups=df[TEMPLATE_GROUP_COL])

    elif protocol == SOURCE_HOLDOUT_PROTOCOL:
        splitter = LeaveOneGroupOut()
        split_iterator = splitter.split(df[TEXT_COL], df[LABEL_COL], groups=df[SOURCE_COL])
        n_splits = int(df[SOURCE_COL].nunique())

    else:
        raise ValueError(f"Unknown CV protocol: {protocol}")

    fold_ids = pd.Series(-1, index=df.index, dtype="int64")

    for fold_idx, (_, held_out_idx) in enumerate(split_iterator):
        fold_ids.iloc[held_out_idx] = fold_idx

    if (fold_ids < 0).any() or fold_ids.nunique() != n_splits:
        raise AssertionError("Every CV row must be assigned to exactly one held-out fold")

    if (
        protocol == TEMPLATE_GROUP_PROTOCOL
        and df.assign(_fold=fold_ids).groupby(TEMPLATE_GROUP_COL)["_fold"].nunique().max() != 1
    ):
        raise AssertionError("A template group crosses CV folds")

    for fold_idx in sorted(fold_ids.unique()):
        held_out_labels = df.loc[fold_ids == fold_idx, LABEL_COL].nunique()
        training_labels = df.loc[fold_ids != fold_idx, LABEL_COL].nunique()

        if training_labels < 2:
            raise ValueError(f"CV fold {fold_idx} does not contain both labels in train and validation")

    return fold_ids


def build_inner_validation_mask(
    outer_training_df: pd.DataFrame,
    *,
    outer_fold: int,
    n_splits: int = 5,
) -> pd.Series:
    """Select a deterministic group-disjoint inner fold for epoch selection."""
    inner_folds = build_cv_folds(
        outer_training_df,
        n_splits=n_splits,
        protocol=TEMPLATE_GROUP_PROTOCOL,
        random_state=RANDOM_STATE + 10_000 + outer_fold,
    )
    selected_fold = outer_fold % int(inner_folds.nunique())
    mask = inner_folds == selected_fold

    if not mask.any() or mask.all():
        raise AssertionError("Inner validation must leave non-empty training and validation subsets")

    train_groups = set(outer_training_df.loc[~mask, TEMPLATE_GROUP_COL].astype(str))
    validation_groups = set(outer_training_df.loc[mask, TEMPLATE_GROUP_COL].astype(str))

    if train_groups & validation_groups:
        raise AssertionError("A template group crosses inner training and validation")

    return mask


def outer_training_mask(df: pd.DataFrame, folds: pd.Series, fold: int) -> pd.Series:
    """Purge cross-generator template relatives from source-holdout training."""
    held_out_groups = set(df.loc[folds == fold, TEMPLATE_GROUP_COL])
    return (folds != fold) & ~df[TEMPLATE_GROUP_COL].isin(held_out_groups)
