"""Single, shared inference implementation for every evaluation stage."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from numpy.typing import NDArray
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import BASE_DIR, DEFAULT_MAX_LENGTH, SAVED_MODELS_DIR
from src.models.provenance import artefact_matches_current_split
from src.utils.logger import get_logger

logger = get_logger(__name__)


def transformer_weights_exist(model_dir: Path) -> bool:
    """Return whether a Hugging Face directory contains actual model weights."""
    return any((model_dir / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))


def registered_model_names() -> List[str]:
    """Read reproducible experiment names; unregistered legacy models are excluded."""
    with (BASE_DIR / "params.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    return [str(experiment["name"]) for experiment in config.get("experiments", [])]


def discover_models(saved_models_dir: Optional[Path] = None, require_complete: bool = True) -> List[Dict[str, Any]]:
    """Discover only registered and complete model artefacts."""
    root = saved_models_dir or SAVED_MODELS_DIR
    models: List[Dict[str, Any]] = []
    baseline_path = root / "baseline" / "pipeline.joblib"

    if baseline_path.exists() and artefact_matches_current_split(root / "baseline"):
        models.append({"name": "baseline", "type": "sklearn", "path": root / "baseline"})

    elif baseline_path.exists():
        logger.warning(f"Skipping stale baseline not tied to current split: {root / 'baseline'}")

    for name in registered_model_names():
        model_dir = root / name

        if transformer_weights_exist(model_dir) and artefact_matches_current_split(model_dir):
            models.append({"name": name, "type": "transformer", "path": model_dir})

        elif transformer_weights_exist(model_dir):
            logger.warning(f"Skipping stale transformer not tied to current split: {model_dir}")

        elif model_dir.exists():
            logger.warning(f"Skipping incomplete model without weights: {model_dir}")

    if require_complete and {m["name"] for m in models} != {"baseline", *registered_model_names()}:
        raise RuntimeError("Final study is incomplete: train baseline and all registered transformers")

    return models


class TransformerPredictor:
    """Reusable model session with length buckets and bounded MPS memory."""

    def __init__(self, model_path: Path, device: Optional[str] = None) -> None:
        from src.models.runtime import configure_runtime

        self.device = device or str(configure_runtime())
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, attn_implementation="eager")
        self.model.to(self.device)
        self.model.eval()

        if self.model.config.num_labels != 2:
            raise ValueError("Expected a binary classifier")

    def predict(self, texts: List[str], batch_size: int = 16) -> NDArray[np.float64]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        if not texts:
            return np.empty(0, dtype=np.float64)

        encoded = self.tokenizer(texts, truncation=True, max_length=DEFAULT_MAX_LENGTH)
        order = sorted(range(len(texts)), key=lambda i: len(encoded["input_ids"][i]))
        result = np.empty(len(texts), dtype=np.float64)
        position = 0

        while position < len(order):
            indices = order[position : position + batch_size]
            inputs = None

            try:
                rows = [{key: value[i] for key, value in encoded.items()} for i in indices]
                inputs = self.tokenizer.pad(rows, padding=True, pad_to_multiple_of=8, return_tensors="pt").to(
                    self.device
                )

                with torch.inference_mode():
                    probabilities = torch.softmax(self.model(**inputs).logits, dim=-1)[:, 1]

                result[indices] = probabilities.cpu().float().numpy()
                del probabilities, inputs
                position += len(indices)

            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch_size == 1:
                    raise

                del inputs
                from src.models.runtime import release_memory

                release_memory()
                batch_size = max(1, batch_size // 2)
                logger.warning(f"Inference memory limit reached; reducing batch to {batch_size}")

        return result

    def close(self) -> None:
        from src.models.runtime import release_memory

        del self.model
        release_memory()


def predict_transformer(
    model_path: Path,
    texts: List[str],
    batch_size: int = 16,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: Optional[str] = None,
) -> NDArray[np.float64]:
    if max_length != DEFAULT_MAX_LENGTH:
        raise ValueError("Non-canonical inference length")

    session = TransformerPredictor(model_path, device)

    try:
        return session.predict(texts, batch_size)

    finally:
        session.close()


def predict_model(model_config: Dict[str, Any], texts: pd.Series) -> NDArray[np.float64]:
    """Cache raw probabilities by exact text, model, implementation and device."""
    from src.config import RESULTS_DIR
    from src.models.runtime import get_device
    from src.utils.artifacts import atomic_json, file_sha256, identity

    model_path = Path(model_config["path"])

    if not artefact_matches_current_split(model_path):
        raise ValueError(f"Stale or corrupt model: {model_path}")

    text_list = texts.astype(str).tolist()
    key = identity(
        {
            "model": file_sha256(model_path / "training_manifest.json"),
            "texts": text_list,
            "length": DEFAULT_MAX_LENGTH,
            "device": str(get_device()),
        }
    )
    cache = RESULTS_DIR / "prediction_cache" / f"{key}.npy"

    if cache.exists():
        metadata = json.loads(cache.with_suffix(".json").read_text())
        if metadata != {"input_identity": key, "sha256": file_sha256(cache)}:
            raise ValueError(f"Prediction cache content changed: {cache}")
        values = np.load(cache, allow_pickle=False)

        if values.shape != (len(texts),) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Invalid prediction cache: {cache}")

        return np.asarray(values, dtype=np.float64)

    if model_config["type"] == "sklearn":
        pipeline = joblib.load(model_path / "pipeline.joblib")
        values = np.asarray(pipeline.predict_proba(texts)[:, 1], dtype=np.float64)

    elif model_config["type"] == "transformer":
        values = predict_transformer(model_path, text_list)

    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")

    cache.parent.mkdir(parents=True, exist_ok=True)

    with cache.with_suffix(".tmp").open("wb") as handle:
        np.save(handle, values)

    cache.with_suffix(".tmp").replace(cache)
    atomic_json(cache.with_suffix(".json"), {"input_identity": key, "sha256": file_sha256(cache)})
    return np.asarray(values, dtype=np.float64)
