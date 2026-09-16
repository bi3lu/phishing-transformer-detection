"""Run versioned, resumable experiments. Set PHISHING_RUN_ID and PHISHING_SEED."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from src.config import BASE_DIR, PROTOCOL_VERSION, RANDOM_STATE, RESULTS_DIR
from src.utils.artifacts import atomic_json, bind_run, file_sha256, identity, stage_status
from src.utils.logger import get_logger

logger = get_logger(__name__)
STEPS = [
    "preprocess",
    "split",
    "baseline",
    "finetune",
    "threshold",
    "ensemble",
    "kfold",
    "source-holdout",
    "evaluate",
    "analysis",
    "report",
]


def get_experiment_names() -> list[str]:
    # The coordinating process must not initialize PyTorch/Metal.
    config = yaml.safe_load((BASE_DIR / "params.yaml").read_text())
    return [str(experiment["name"]) for experiment in config["experiments"]]


def completed_folds(directory: Path) -> set[str]:
    return {p.parent.name for p in directory.glob("fold_*/completed.json")}


def identity_differences(saved: dict[str, Any], current: dict[str, Any], prefix: str = "") -> list[str]:
    differences = []
    for key in sorted(saved.keys() | current.keys()):
        name = f"{prefix}.{key}" if prefix else key
        before, after = saved.get(key), current.get(key)

        if isinstance(before, dict) and isinstance(after, dict):
            differences.extend(identity_differences(before, after, name))

        elif before != after:
            differences.append(f"{name}: saved={before!r}; current={after!r}")

    return differences


def check_run_identity() -> None:
    """Explain a mismatch without rewriting historical manifests or loading models."""
    from src.utils.artifacts import environment_metadata, implementation_identity

    path = RESULTS_DIR / "run_manifest.json"

    if not path.exists():
        return

    saved = json.loads(path.read_text())
    current = {
        "protocol": PROTOCOL_VERSION,
        "seed": RANDOM_STATE,
        "implementation": implementation_identity(),
        "environment": environment_metadata(),
    }
    differences = identity_differences(saved, current)
    if differences and any(
        p.is_file() and p.name not in {"run_manifest.json", "params.yaml"} for p in RESULTS_DIR.rglob("*")
    ):
        raise ValueError(
            f"Cannot resume {RESULTS_DIR}:\n  "
            + "\n  ".join(differences)
            + "\nExisting results are preserved. Restore the original environment, or use a new "
            "PHISHING_RUN_ID and run all training/evaluation stages (not only CV). "
            "The existing data split can be reused with --skip preprocess split."
        )


def run_isolated(step: str, experiment: str | None, lock_fd: int) -> None:
    """Release the entire Metal process between models/protocols, retaining folds.

    A CV OOM may be retried only if this attempt completed new folds. Thus an
    oversized individual fold fails visibly instead of looping indefinitely.
    """
    protocol = {"kfold": "template_group", "source-holdout": "source_holdout"}.get(step)
    cv_directory = RESULTS_DIR / "cross_validation" / str(experiment) / str(protocol)
    command = [sys.executable, str(BASE_DIR / "main.py"), "--only", step, "--worker-lock-fd", str(lock_fd)]

    if experiment is not None:
        command += ["--experiments", experiment]

    while True:
        before = completed_folds(cv_directory) if protocol else set()
        started = datetime.now(timezone.utc)
        logger.info(f"Fresh process: {step} / {experiment or 'all'}")
        result = subprocess.run(command, cwd=BASE_DIR, pass_fds=(lock_fd,), check=False)
        after = completed_folds(cv_directory) if protocol else set()
        atomic_json(
            RESULTS_DIR / "execution" / f"{started.strftime('%Y%m%dT%H%M%S%f')}-{step}-{experiment or 'all'}.json",
            {
                "isolation": "one_process_per_model_and_protocol_or_evaluation_stage",
                "launcher_sha256": file_sha256(BASE_DIR / "main.py"),
                "step": step,
                "experiment": experiment,
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "returncode": result.returncode,
                "completed_folds_before": sorted(before),
                "completed_folds_after": sorted(after),
            },
        )
        if result.returncode == 0:
            return

        status_path = cv_directory / "status.json"
        status = json.loads(status_path.read_text()) if protocol and status_path.exists() else {}

        if (
            result.returncode > 0
            and status.get("status") == "failed"
            and "MPS backend out of memory" in status.get("error", "")
            and after > before
        ):
            logger.warning("CV reached an MPS OOM after completed folds; resuming remaining folds in a fresh process")
            continue

        raise subprocess.CalledProcessError(result.returncode, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=STEPS)
    parser.add_argument("--skip", nargs="+", choices=STEPS, default=[])
    parser.add_argument("--experiments", nargs="+", default=[])
    parser.add_argument(
        "--check-run", action="store_true", help="Check run identity without training or writing results"
    )
    parser.add_argument("--worker-lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_run:
        check_run_identity()
        logger.info(f"Run identity is compatible, or this is a new run: {RESULTS_DIR}")
        return
    experiments = args.experiments or get_experiment_names()

    allowed = set(get_experiment_names()) | ({"baseline"} if args.worker_lock_fd is not None else set())
    if set(experiments) - allowed:
        parser.error("Unknown experiment name")

    steps = [s for s in STEPS if s in args.only] if args.only else [s for s in STEPS if s not in args.skip]

    if args.worker_lock_fd is not None and (len(steps) != 1 or len(args.experiments) > 1):
        parser.error("An isolated worker accepts one stage and at most one model")
    lock_path = BASE_DIR / "results" / ".pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if args.worker_lock_fd is not None:
        inherited = os.fstat(args.worker_lock_fd)
        expected = lock_path.stat()
        if (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("Worker did not inherit the pipeline lock")
        fcntl.flock(args.worker_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute_steps(steps, experiments, args.worker_lock_fd, worker=True)
    else:
        with lock_path.open("a+") as pipeline_lock:
            try:
                fcntl.flock(pipeline_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another pipeline is running; unified GPU memory is shared") from exc
            execute_steps(steps, experiments, pipeline_lock.fileno(), worker=False)


def execute_steps(steps: list[str], experiments: list[str], lock_fd: int, *, worker: bool) -> None:
    if worker:
        logger.info(f"Isolated worker PID: {os.getpid()}")

        if any(step not in ("preprocess", "split") for step in steps):
            check_run_identity()

    for step in steps:
        logger.info(f"Stage: {step}; results: {RESULTS_DIR}")

        if not worker and step not in ("preprocess", "split"):
            if step == "finetune":
                for experiment in experiments:
                    run_isolated(step, experiment, lock_fd)

            elif step in ("kfold", "source-holdout"):
                for experiment in ["baseline", *experiments]:
                    run_isolated(step, experiment, lock_fd)

            else:
                run_isolated(step, None, lock_fd)

            continue

        if step == "preprocess":
            from src.data.preprocess_data import main as run

            run()

        elif step == "split":
            from src.data.split_data import main as run

            run()

        elif step == "baseline":
            from src.models.baseline import main as run

            run()

        elif step == "finetune":
            from src.models.fine_tune import main as finetune

            for experiment in experiments:
                finetune(experiment)

        elif step in ("kfold", "source-holdout"):
            from src.models.kfold_cv import run_kfold

            for experiment in experiments:
                run_kfold(experiment, protocol="template_group" if step == "kfold" else "source_holdout")

        else:
            from src.evaluation import analysis, ensemble, evaluate, threshold_analysis
            from src.evaluation.report import check_completeness

            if step == "report":
                status = check_completeness()
                atomic_json(RESULTS_DIR / "study_status.json", status)

                if not status["complete"]:
                    raise RuntimeError(json.dumps(status, indent=2))

                continue

            functions: dict[str, Callable[[], None]] = {
                "threshold": threshold_analysis.main,
                "ensemble": ensemble.main,
                "evaluate": evaluate.main,
                "analysis": analysis.main,
            }
            bind_run()

            with stage_status(RESULTS_DIR / "stages" / step, identity((RESULTS_DIR / "run_manifest.json").read_text())):
                functions[step]()

    logger.info("Requested stages completed")


if __name__ == "__main__":
    main()
