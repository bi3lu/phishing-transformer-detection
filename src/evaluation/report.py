"""Check study completeness and summarize seed variation without test selection."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import BASE_DIR, RESULTS_DIR
from src.evaluation.inference import registered_model_names
from src.utils.artifacts import atomic_json, file_sha256, verify_run


def check_completeness(root: Path = RESULTS_DIR) -> dict[str, Any]:
    missing = []

    for model in ["baseline", *registered_model_names()]:
        for protocol in ("template_group", "source_holdout"):
            directory = root / "cross_validation" / model / protocol

            for name in (
                "summary.json",
                "fold_metrics.csv",
                "oof_predictions.csv",
                "oof_group_bootstrap_ci.csv",
                "oof_group_binomial_ci.csv",
            ):
                if not (directory / name).exists():
                    missing.append(str((directory / name).relative_to(root)))

            status = directory / "status.json"

            if not status.exists() or json.loads(status.read_text()).get("status") != "completed":
                missing.append(str(status.relative_to(root)) + ": not completed")

            manifest_path = directory / "cv_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                for fold in range(1, int(manifest["folds"]) + 1):
                    fold_dir = directory / f"fold_{fold}"
                    completed = fold_dir / "completed.json"
                    if not completed.exists():
                        missing.append(str(completed.relative_to(root)))
                        continue
                    record = json.loads(completed.read_text())
                    for filename, digest in record["files"].items():
                        path = fold_dir / filename
                        if not path.exists() or file_sha256(path) != digest:
                            missing.append(str(path.relative_to(root)) + ": corrupt")

    for name in (
        "final_test_metrics.csv",
        "final_test_predictions.csv",
        "final_test_manifest.json",
        "final_test_slice_metrics.csv",
        "final_test_group_bootstrap_ci.csv",
        "final_test_group_binomial_ci.csv",
        "mcnemar_tests.csv",
        "error_analysis.csv",
        "threshold_selection/thresholds.json",
        "ensemble_selection/ensemble_config.json",
    ):
        if not (root / name).exists():
            missing.append(name)

    for step in ("threshold", "ensemble", "evaluate", "analysis"):
        status = root / "stages" / step / "status.json"
        if not status.exists() or json.loads(status.read_text()).get("status") != "completed":
            missing.append(f"stages/{step}: not completed")
    if not missing and root == RESULTS_DIR:
        from src.evaluation.evaluate import load_final_predictions

        verify_run()
        load_final_predictions()
    return {
        "complete": not missing,
        "completion_scope": "computational_artifacts_only",
        "human_label_review": "pending; see label_review_queue.csv",
        "confirmatory_study_ready": False,
        "missing": missing,
        "test_status": "exploratory_reanalysis_previously_inspected_corpus",
    }


def summarize_seeds(run_root: Path) -> pd.DataFrame:
    frames = []
    expected_implementation = None
    for directory in sorted(run_root.glob("seed-*")):
        run_manifest_path = directory / "run_manifest.json"
        if not run_manifest_path.exists():
            continue
        run_manifest = json.loads(run_manifest_path.read_text())
        if expected_implementation is not None and run_manifest["implementation"] != expected_implementation:
            raise ValueError("Cannot aggregate different implementations/configurations across seeds")
        expected_implementation = run_manifest["implementation"]
        for path in sorted((directory / "cross_validation").glob("*/*/fold_metrics.csv")):
            status = path.parent / "status.json"

            if not status.exists() or json.loads(status.read_text()).get("status") != "completed":
                continue

            frame = pd.read_csv(path)

            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                frames.append(
                    {
                        "seed": directory.name,
                        "model": path.parent.parent.name,
                        "protocol": path.parent.name,
                        "metric": metric,
                        "value": float(frame[metric].mean()),
                    }
                )
    if not frames:
        raise ValueError("No completed CV runs available")

    data = pd.DataFrame(frames)
    return data.groupby(["model", "protocol", "metric"]).value.agg(["mean", "std", "count"]).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--across-seeds", action="store_true")
    args = parser.parse_args()

    if args.across_seeds:
        output = summarize_seeds(RESULTS_DIR.parent)
        output.to_csv(RESULTS_DIR.parent / "seed_variation.csv", index=False)
        print(output.to_string(index=False))

    else:
        status = check_completeness()
        atomic_json(RESULTS_DIR / "study_status.json", status)
        print(json.dumps(status, indent=2))

        if not status["complete"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
