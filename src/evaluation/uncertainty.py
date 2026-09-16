"""Uncertainty estimates that respect template-family dependence."""

from typing import Dict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import beta
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

BOOTSTRAP_METRICS = ("accuracy", "f1", "precision", "recall", "roc_auc")


def _clopper_pearson(successes: int, trials: int, confidence_level: float) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")

    alpha = 1.0 - confidence_level
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return lower, upper


def binary_classification_metrics(
    labels: NDArray[np.int_], probabilities: NDArray[np.floating], threshold: float = 0.5
) -> Dict[str, float]:
    """Return the canonical binary metrics for probabilities and a fixed threshold."""
    predicted = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "roc_auc": float("nan"),
    }
    if np.unique(labels).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))

    return metrics


def grouped_bootstrap_confidence_intervals(
    labels: NDArray[np.int_],
    probabilities: NDArray[np.floating],
    groups: NDArray[np.str_],
    *,
    threshold: float = 0.5,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Percentile cluster bootstrap, resampling whole ``Template_Group`` units.

    Repeated rows from one template family are never split across bootstrap
    units. This avoids falsely treating within-template variants as independent
    observations.
    """
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    groups = np.asarray(groups, dtype=str)

    if not (len(labels) == len(probabilities) == len(groups)):
        raise ValueError("labels, probabilities, and groups must have equal lengths")

    if len(labels) == 0:
        raise ValueError("cannot bootstrap an empty sample")

    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    unique_groups, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(random_state)
    samples: Dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    predicted = probabilities >= threshold
    positive = labels == 1
    order = np.argsort(probabilities, kind="stable")
    sorted_probs = probabilities[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_probs)) + 1]

    for _ in range(n_resamples):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        weights = np.bincount(sampled, minlength=len(unique_groups))[inverse]
        tp = float(weights[positive & predicted].sum())
        fp = float(weights[~positive & predicted].sum())
        fn = float(weights[positive & ~predicted].sum())
        tn = float(weights[~positive & ~predicted].sum())
        metrics = {
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }

        pos = np.add.reduceat((weights * positive)[order], starts)
        neg = np.add.reduceat((weights * ~positive)[order], starts)

        if pos.sum() and neg.sum():
            metrics["roc_auc"] = float(np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (pos.sum() * neg.sum()))

        for metric, value in metrics.items():
            samples[metric].append(value)

    alpha = 1.0 - confidence_level
    estimates = binary_classification_metrics(labels, probabilities, threshold)
    intervals: Dict[str, Dict[str, float]] = {}

    for metric in BOOTSTRAP_METRICS:
        values = np.asarray(samples[metric], dtype=float)

        if values.size == 0:
            lower = upper = float("nan")

        else:
            lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])

        intervals[metric] = {
            "estimate": float(estimates[metric]),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "confidence_level": float(confidence_level),
            "successful_resamples": int(values.size),
            "template_groups": int(len(unique_groups)),
            "degenerate": bool(values.size and np.ptp(values) == 0),
        }

    return intervals


def template_group_binomial_confidence_intervals(
    labels: NDArray[np.int_],
    probabilities: NDArray[np.floating],
    groups: NDArray[np.str_],
    *,
    threshold: float = 0.5,
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Exact intervals after reducing data to independent template families.

    These intervals complement the cluster bootstrap when no observed errors
    would otherwise produce a misleading degenerate interval at one.
    """
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    groups = np.asarray(groups, dtype=str)

    if not (len(labels) == len(probabilities) == len(groups)):
        raise ValueError("labels, probabilities, and groups must have equal lengths")

    if not 0 < confidence_level < 1 or len(labels) == 0:
        raise ValueError("Invalid confidence level or empty data")

    frame = pd.DataFrame({"group": groups, "correct": (probabilities >= threshold).astype(int) == labels})
    grouped = frame.groupby("group")["correct"].all()
    counts = {"group_all_correct": (int(grouped.sum()), len(grouped))}
    intervals: Dict[str, Dict[str, float]] = {}

    for metric, (successes, trials) in counts.items():
        lower, upper = _clopper_pearson(successes, trials, confidence_level)
        intervals[metric] = {
            "estimate": float(successes / trials) if trials else float("nan"),
            "ci_lower": lower,
            "ci_upper": upper,
            "confidence_level": float(confidence_level),
            "successes": int(successes),
            "trials": int(trials),
            "template_groups": int(len(grouped)),
        }

    return intervals
