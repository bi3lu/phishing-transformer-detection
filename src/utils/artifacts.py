"""Atomic artefacts and content identities for resumable experiments."""

import fcntl
import hashlib
import importlib.metadata
import json
import platform
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.config import BASE_DIR, PROTOCOL_VERSION, RANDOM_STATE, RESULTS_DIR


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def identity(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def implementation_identity() -> str:
    paths = sorted((BASE_DIR / "src").rglob("*.py")) + [BASE_DIR / "params.yaml", BASE_DIR / "uv.lock"]
    paths += [BASE_DIR / "docs" / name for name in ("data_sources.json", "template_overrides.json")]
    return identity({str(p.relative_to(BASE_DIR)): file_sha256(p) for p in paths})


def environment_metadata() -> dict[str, Any]:
    from src.models.runtime import get_device

    packages = ("torch", "transformers", "accelerate", "numpy", "scikit-learn", "shap", "datasets")
    return {
        "device": str(get_device()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def bind_run() -> None:
    """Refuse to mix code/config/seeds in one named run."""
    path = RESULTS_DIR / "run_manifest.json"
    payload = {
        "protocol": PROTOCOL_VERSION,
        "seed": RANDOM_STATE,
        "implementation": implementation_identity(),
        "environment": environment_metadata(),
    }

    if path.exists() and json.loads(path.read_text()) != payload:
        if any(p.is_file() and p.name not in {"run_manifest.json", "params.yaml"} for p in RESULTS_DIR.rglob("*")):
            raise ValueError("Run identity changed. Use a new PHISHING_RUN_ID; existing results are preserved.")

    atomic_json(path, payload)
    (RESULTS_DIR / "params.yaml").write_bytes((BASE_DIR / "params.yaml").read_bytes())


def verify_run() -> None:
    """Read-only identity check for notebooks and report consumers."""
    payload = json.loads((RESULTS_DIR / "run_manifest.json").read_text())
    if payload != {
        "protocol": PROTOCOL_VERSION,
        "seed": RANDOM_STATE,
        "implementation": implementation_identity(),
        "environment": environment_metadata(),
    }:
        raise ValueError("Run identity differs from current code/config/environment")


@contextmanager
def stage_status(directory: Path, fingerprint: str) -> Iterator[None]:
    """Record failure/interruption explicitly; completion is written last."""
    directory.mkdir(parents=True, exist_ok=True)

    with (directory / ".stage.lock").open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        except BlockingIOError as exc:
            raise RuntimeError(f"Stage already running: {directory}") from exc

        path = directory / "status.json"

        if path.exists() and json.loads(path.read_text()).get("fingerprint") != fingerprint:
            raise ValueError(f"Stale stage at {directory}; choose a new run ID")

        payload = {
            "fingerprint": fingerprint,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        atomic_json(path, payload)
        try:
            yield

        except BaseException as exc:
            atomic_json(path, {**payload, "status": "failed", "error": str(exc)})
            raise

        else:
            atomic_json(path, {**payload, "status": "completed"})
