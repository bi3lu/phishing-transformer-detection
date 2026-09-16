"""Resumable raw-model robustness CV; test is never a selection input.

Every model is fitted on all outer-training rows (after source-template purge).
Transformer epochs are chosen inside that training set, then weights are reset
for the full refit. This evaluates raw models at 0.5, not calibrated deployment.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.config import (
    BASE_DIR,
    LABEL_COL,
    PROTOCOL_VERSION,
    RANDOM_STATE,
    RESULTS_DIR,
    SOURCE_COL,
    TEMPLATE_GROUP_COL,
    TEXT_COL,
)
from src.data.cv_folds import SOURCE_HOLDOUT_PROTOCOL, TEMPLATE_GROUP_PROTOCOL, build_cv_folds, outer_training_mask
from src.data.load_data import load_split
from src.evaluation.uncertainty import (
    binary_classification_metrics,
    grouped_bootstrap_confidence_intervals,
    template_group_binomial_confidence_intervals,
)
from src.models.baseline import build_baseline_pipeline
from src.models.fine_tune import prepare_dataset, select_epoch, train_phase
from src.models.provenance import current_split_manifest
from src.models.runtime import release_memory
from src.utils.artifacts import atomic_json, bind_run, file_sha256, identity, stage_status

CV_PROTOCOL_VERSION = PROTOCOL_VERSION


def load_cv_data() -> pd.DataFrame:
    return pd.concat([load_split(s).assign(CV_Origin_Split=s) for s in ("train", "val")], ignore_index=True)


def run_kfold(experiment_name: str, n_splits: int = 5, protocol: str = TEMPLATE_GROUP_PROTOCOL) -> pd.DataFrame:
    bind_run()
    config = yaml.safe_load((BASE_DIR / "params.yaml").read_text())
    cv_config = config["research_protocol"]["cross_validation"]
    experiment = next((e for e in config["experiments"] if e["name"] == experiment_name), None)

    if experiment_name != "baseline" and experiment is None:
        raise ValueError(f"Unknown experiment {experiment_name}")

    if n_splits != cv_config["folds"]:
        raise ValueError("Use the registered number of folds")

    df = load_cv_data()
    folds = build_cv_folds(df, n_splits, protocol)
    output = RESULTS_DIR / "cross_validation" / experiment_name / protocol
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": experiment_name,
        "protocol": protocol,
        "evaluation_target": "raw_model_refitted_on_all_outer_training",
        "threshold": 0.5,
        "calibration": "none",
        "test_loaded": False,
        "seed": RANDOM_STATE,
        "data": current_split_manifest(),
        "config": experiment or config["baseline"],
        "folds": int(folds.nunique()),
        "rows": len(df),
    }
    fingerprint = identity(manifest)
    output.mkdir(parents=True, exist_ok=True)
    assignments = df[["Record_ID", TEMPLATE_GROUP_COL, SOURCE_COL, LABEL_COL, "CV_Origin_Split"]].copy()
    assignments["outer_fold"] = folds + 1
    old_manifest = output / "cv_manifest.json"

    if old_manifest.exists() and json.loads(old_manifest.read_text()) != manifest:
        raise ValueError("CV inputs changed; choose a new run ID")

    atomic_json(old_manifest, manifest)
    assignments.to_csv(output / "fold_assignments.csv", index=False)
    metrics_rows, prediction_frames = [], []

    with stage_status(output, fingerprint):
        for fold in sorted(folds.unique()):
            fold_dir = output / f"fold_{fold + 1}"
            done = fold_dir / "completed.json"

            if done.exists():
                saved = json.loads(done.read_text())

                if saved["fingerprint"] != fingerprint or any(
                    file_sha256(fold_dir / name) != digest for name, digest in saved["files"].items()
                ):
                    raise ValueError(f"Stale/corrupt completed CV fold: {fold_dir}")

                metrics_rows.append(json.loads((fold_dir / "metrics.json").read_text()))
                prediction_frames.append(pd.read_csv(fold_dir / "predictions.csv"))
                continue

            training = df.loc[outer_training_mask(df, folds, int(fold))].copy()
            evaluation = df.loc[folds == fold].copy()
            fold_dir.mkdir(parents=True, exist_ok=True)
            roles = assignments.copy()
            roles["role"] = "purged_template_relative"
            roles.loc[training.index, "role"] = "outer_train_refit"
            roles.loc[evaluation.index, "role"] = "outer_evaluation"
            roles.to_csv(fold_dir / "assignments.csv", index=False)

            with stage_status(fold_dir, fingerprint):
                selection: dict[str, Any] = {}

                if experiment_name == "baseline":
                    model = build_baseline_pipeline(config)
                    model.fit(training[TEXT_COL], training[LABEL_COL].astype(int))
                    probs = model.predict_proba(evaluation[TEXT_COL])[:, 1]
                    del model

                else:
                    assert experiment is not None
                    selection = select_epoch(
                        experiment, training, fold_dir, RANDOM_STATE + int(fold), selection_fold=int(fold)
                    )
                    trainer, tokenizer = train_phase(
                        experiment,
                        training,
                        None,
                        fold_dir / "refit",
                        RANDOM_STATE + int(fold),
                        selection["selected_epoch"],
                    )
                    output_predictions = trainer.predict(
                        prepare_dataset(evaluation, tokenizer, int(experiment["max_length"]))
                    )
                    probs = torch.softmax(torch.tensor(output_predictions.predictions), dim=-1)[:, 1].numpy()
                    del trainer, tokenizer
                    release_memory()

                labels = evaluation[LABEL_COL].to_numpy(dtype=int)
                row = {
                    "fold": int(fold) + 1,
                    "held_out_source": (
                        str(evaluation[SOURCE_COL].iloc[0]) if protocol == SOURCE_HOLDOUT_PROTOCOL else ""
                    ),
                    "outer_train_rows": len(training),
                    "refit_training_rows": len(training),
                    "outer_validation_rows": len(evaluation),
                    "purged_rows": int((folds != fold).sum()) - len(training),
                    "selected_epoch": selection.get("selected_epoch"),
                    **binary_classification_metrics(labels, probs),
                }
                predictions = assignments.loc[evaluation.index].copy()
                predictions.insert(0, "row_position", evaluation.index)
                predictions["probability"] = probs
                predictions["prediction"] = (probs >= 0.5).astype(int)
                predictions.to_csv(fold_dir / "predictions.csv", index=False)
                atomic_json(fold_dir / "metrics.json", row)
                atomic_json(
                    done,
                    {
                        "fingerprint": fingerprint,
                        "files": {name: file_sha256(fold_dir / name) for name in ("metrics.json", "predictions.csv")},
                    },
                )
                metrics_rows.append(row)
                prediction_frames.append(predictions)

        metrics = pd.DataFrame(metrics_rows)
        oof = pd.concat(prediction_frames, ignore_index=True).sort_values("row_position")

        if len(oof) != len(df) or oof.row_position.nunique() != len(df) or set(oof.Record_ID) != set(df.Record_ID):
            raise ValueError("Incomplete/duplicate OOF predictions")

        metrics.to_csv(output / "fold_metrics.csv", index=False)
        oof.to_csv(output / "oof_predictions.csv", index=False)
        labels, probabilities, groups = (
            oof[LABEL_COL].to_numpy(dtype=int),
            oof.probability.to_numpy(),
            oof[TEMPLATE_GROUP_COL].to_numpy(dtype=str),
        )
        intervals = grouped_bootstrap_confidence_intervals(
            labels, probabilities, groups, n_resamples=cv_config["bootstrap_resamples"]
        )
        binomial = template_group_binomial_confidence_intervals(labels, probabilities, groups)
        pd.DataFrame([{"metric": key, **value} for key, value in intervals.items()]).to_csv(
            output / "oof_group_bootstrap_ci.csv", index=False
        )
        pd.DataFrame([{"metric": key, **value} for key, value in binomial.items()]).to_csv(
            output / "oof_group_binomial_ci.csv", index=False
        )
        atomic_json(
            output / "summary.json",
            {
                **manifest,
                "pooled_oof_metrics": binary_classification_metrics(labels, probabilities),
                "fold_distribution": {
                    metric: {"mean": float(metrics[metric].mean()), "std": float(metrics[metric].std())}
                    for metric in ("f1", "precision", "recall", "accuracy", "roc_auc")
                },
                "group_bootstrap_confidence_intervals": intervals,
                "uncertainty_scope": "conditional_on_fitted_models; excludes_training_seed_variation",
            },
        )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument(
        "--protocol", choices=[TEMPLATE_GROUP_PROTOCOL, SOURCE_HOLDOUT_PROTOCOL], default=TEMPLATE_GROUP_PROTOCOL
    )
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()
    run_kfold(args.experiment_name, args.n_splits, args.protocol)


if __name__ == "__main__":
    main()
