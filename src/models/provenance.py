"""Verify physical data, code and model artefacts before reuse."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import BASE_DIR, PROTOCOL_VERSION, RANDOM_STATE, SPLIT_DATA_DIR
from src.utils.artifacts import atomic_json, environment_metadata, file_sha256, implementation_identity

TRAINING_MANIFEST = "training_manifest.json"


def current_split_manifest(verify_splits: tuple[str, ...] = ("train", "val")) -> dict[str, Any]:
    path = SPLIT_DATA_DIR / "split_manifest.json"
    payload: dict[str, Any] = json.loads(path.read_text())

    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Legacy split protocol; regenerate v2 preprocessing and splits")

    for name in verify_splits:
        if file_sha256(SPLIT_DATA_DIR / f"{name}.csv") != payload["splits"][name]["sha256"]:
            raise ValueError(f"Physical {name}.csv differs from split manifest")

    source = Path(payload["source_dataset"])
    source = source if source.is_absolute() else BASE_DIR / source

    if file_sha256(source) != payload["source_sha256"]:
        raise ValueError("Canonical dataset differs from split manifest")

    return payload


def model_file_hashes(model_dir: Path) -> dict[str, str]:
    return {
        p.name: file_sha256(p)
        for p in sorted(model_dir.iterdir())
        if p.is_file()
        and p.name not in {TRAINING_MANIFEST, "status.json", ".stage.lock"}
        and not p.name.endswith(".tmp")
    }


def write_training_manifest(model_dir: Path, model_name: str, parameters: dict[str, Any]) -> None:
    split = current_split_manifest()
    payload = {
        "protocol": PROTOCOL_VERSION,
        "model": model_name,
        "status": "completed",
        "seed": RANDOM_STATE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": implementation_identity(),
        "environment": environment_metadata(),
        "source_dataset_sha256": split["source_sha256"],
        "train_sha256": split["splits"]["train"]["sha256"],
        "validation_sha256": split["splits"]["val"]["sha256"],
        "parameters": parameters,
        "files": model_file_hashes(model_dir),
    }
    atomic_json(model_dir / TRAINING_MANIFEST, payload)


def artefact_matches_current_split(model_dir: Path) -> bool:
    path = model_dir / TRAINING_MANIFEST

    if not path.exists():
        return False

    trained = json.loads(path.read_text())
    current = current_split_manifest()

    return bool(
        trained.get("protocol") == PROTOCOL_VERSION
        and trained.get("status") == "completed"
        and trained.get("implementation") == implementation_identity()
        and trained.get("seed") == RANDOM_STATE
        and trained.get("environment") == environment_metadata()
        and trained.get("train_sha256") == current["splits"]["train"]["sha256"]
        and trained.get("validation_sha256") == current["splits"]["val"]["sha256"]
        and trained.get("source_dataset_sha256") == current["source_sha256"]
        and bool(trained.get("files"))
        and trained["files"] == model_file_hashes(model_dir)
    )
