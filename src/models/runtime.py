"""Conservative Apple Silicon runtime; one model in unified memory at a time."""

import gc
import os
from typing import Any

import torch
import yaml
from transformers import enable_full_determinism

from src.config import BASE_DIR, RANDOM_STATE


def get_device() -> torch.device:
    requested = os.environ.get("PHISHING_TRAIN_DEVICE", "auto").lower()

    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cpu" or requested == "mps" and torch.backends.mps.is_available():
        return torch.device(requested)

    if requested == "cuda" and torch.cuda.is_available():
        return torch.device(requested)

    raise ValueError(f"Unavailable PHISHING_TRAIN_DEVICE={requested}")


def configure_runtime(seed: int = RANDOM_STATE) -> torch.device:
    settings = yaml.safe_load((BASE_DIR / "params.yaml").read_text())["hardware"]
    if settings["precision"] != "float32" or settings["attention_implementation"] != "eager":
        raise ValueError("Only the verified float32/eager runtime profile is supported")
    fraction = float(settings["mps_memory_fraction"])
    if not 0 < fraction <= 1:
        raise ValueError("MPS memory fraction must be in (0, 1]")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    enable_full_determinism(seed)
    torch.set_num_threads(min(int(settings["cpu_threads"]), os.cpu_count() or 1))
    device = get_device()
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(fraction)

    return device


def release_memory() -> None:
    gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def runtime_metadata() -> dict[str, Any]:
    device = get_device()
    return {
        "device": str(device),
        "precision": "float32",
        "mps_recommended_bytes": torch.mps.recommended_max_memory() if device.type == "mps" else None,
        "mps_memory_fraction": (
            yaml.safe_load((BASE_DIR / "params.yaml").read_text())["hardware"]["mps_memory_fraction"]
            if device.type == "mps"
            else None
        ),
        "cpu_threads": torch.get_num_threads(),
    }
