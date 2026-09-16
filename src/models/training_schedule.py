"""Memory-aware training schedules with a fixed effective batch size."""

import math
from typing import Any, Dict, Tuple

CANONICAL_EFFECTIVE_BATCH_SIZE = 16


def resolve_training_schedule(experiment_config: Dict[str, Any]) -> Tuple[int, int, int, bool]:
    """Resolve a memory-aware schedule while preserving effective batch size."""
    train_batch_size = int(experiment_config["batch_size"])
    eval_batch_size = int(experiment_config.get("eval_batch_size", train_batch_size))
    accumulation_steps = int(experiment_config.get("gradient_accumulation_steps", 4))
    gradient_checkpointing = bool(experiment_config.get("gradient_checkpointing", False))

    if min(train_batch_size, eval_batch_size, accumulation_steps) < 1:
        raise ValueError("Batch sizes and gradient accumulation must be positive")

    effective_batch_size = train_batch_size * accumulation_steps

    if effective_batch_size != CANONICAL_EFFECTIVE_BATCH_SIZE:
        raise ValueError(
            f"Effective batch size must be {CANONICAL_EFFECTIVE_BATCH_SIZE}, got "
            f"{train_batch_size} * {accumulation_steps} = {effective_batch_size}"
        )

    return train_batch_size, eval_batch_size, accumulation_steps, gradient_checkpointing


def calculate_warmup_steps(
    training_rows: int,
    train_batch_size: int,
    accumulation_steps: int,
    epochs: int,
    warmup_fraction: float = 0.1,
) -> int:
    """Convert the deprecated warmup ratio into deterministic update steps."""
    batches_per_epoch = math.ceil(training_rows / train_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / accumulation_steps)
    return round(updates_per_epoch * epochs * warmup_fraction)
