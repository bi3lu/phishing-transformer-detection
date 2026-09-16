"""Inner epoch selection followed by a fresh refit on all training groups."""

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from src.config import BASE_DIR, DEFAULT_MAX_LENGTH, LABEL_COL, RANDOM_STATE, RESULTS_DIR, SAVED_MODELS_DIR, TEXT_COL
from src.data.cv_folds import build_inner_validation_mask
from src.data.load_data import load_split
from src.evaluation.uncertainty import binary_classification_metrics
from src.models.provenance import artefact_matches_current_split, current_split_manifest, write_training_manifest
from src.models.runtime import configure_runtime, get_device, release_memory, runtime_metadata
from src.models.training_schedule import calculate_warmup_steps, resolve_training_schedule
from src.utils.artifacts import atomic_json, bind_run, identity, stage_status

DEFAULT_FREEZE_LAYERS = 0
DEFAULT_LABEL_SMOOTHING = 0.1


def freeze_lower_layers(model: PreTrainedModel, num_layers_to_freeze: int = 0) -> None:
    if num_layers_to_freeze <= 0:
        return

    for name, parameter in model.named_parameters():
        if "embeddings" in name or any(f"layer.{i}." in name for i in range(num_layers_to_freeze)):
            parameter.requires_grad = False


def prepare_dataset(df: pd.DataFrame, tokenizer: Any, max_length: int) -> Dataset:
    frame = pd.DataFrame({TEXT_COL: df[TEXT_COL].astype(str), "labels": df[LABEL_COL].astype(int)})
    dataset = Dataset.from_pandas(frame, preserve_index=False)
    return dataset.map(
        lambda examples: tokenizer(examples[TEXT_COL], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=[TEXT_COL],
        keep_in_memory=True,
    )


def compute_metrics(pred: Any) -> dict[str, float]:
    probabilities = torch.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1].numpy()
    return binary_classification_metrics(np.asarray(pred.label_ids, dtype=int), probabilities)


class WeightedTrainer(Trainer):
    """Mean weighted loss per example over the *whole* effective batch.

    Weights are fixed inverse training frequencies. Unlike per-microbatch
    weighted means, this objective is invariant to microbatch composition.
    """

    def __init__(self, class_weights: torch.Tensor, label_smoothing: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.model_accepts_loss_kwargs = True

        if self.args.world_size != 1:
            raise ValueError("This verified weighted-loss protocol supports one training device")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> Any:
        labels = inputs["labels"].long()
        outputs = model(**{key: value for key, value in inputs.items() if key != "labels"})
        logits = outputs["logits"]
        weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        loss = nn.functional.cross_entropy(
            logits, labels, weight=weights, label_smoothing=self.label_smoothing, reduction="sum"
        )
        denominator = num_items_in_batch if num_items_in_batch is not None else labels.numel()
        loss = loss / torch.as_tensor(denominator, device=loss.device, dtype=loss.dtype)
        return (loss, outputs) if return_outputs else loss


def train_phase(
    config: dict[str, Any],
    training: pd.DataFrame,
    validation: pd.DataFrame | None,
    output_dir: Path,
    seed: int,
    epochs: int,
) -> tuple[WeightedTrainer, Any]:
    device = configure_runtime(seed)
    batch, eval_batch, accumulation, checkpointing = resolve_training_schedule(config)

    if int(config["max_length"]) != DEFAULT_MAX_LENGTH:
        raise ValueError("Non-canonical maximum sequence length")

    revision = config.get("revision")

    if not revision or (not Path(config["model_name"]).is_dir() and not re.fullmatch(r"[0-9a-f]{40}", str(revision))):
        raise ValueError("Pin an immutable model/tokenizer revision in params.yaml")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=revision)
    training_dataset = prepare_dataset(training, tokenizer, DEFAULT_MAX_LENGTH)
    validation_dataset = prepare_dataset(validation, tokenizer, DEFAULT_MAX_LENGTH) if validation is not None else None

    # Tokenization must not influence random initialization:
    configure_runtime(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], revision=revision, num_labels=2, attn_implementation="eager"
    )
    model.config.id2label = {0: "LEGIT", 1: "PHISH"}
    model.config.label2id = {"LEGIT": 0, "PHISH": 1}
    freeze_lower_layers(model, int(config.get("freeze_layers", 0)))
    counts = training[LABEL_COL].astype(int).value_counts()

    if set(counts.index) != {0, 1}:
        raise ValueError("Training requires both labels")

    weights = torch.tensor([len(training) / (2 * counts[c]) for c in (0, 1)], dtype=torch.float32)
    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch" if validation is not None else "no",
        save_strategy="best" if validation is not None else "no",
        save_total_limit=1,
        save_only_model=True,
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=eval_batch,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=calculate_warmup_steps(len(training), batch, accumulation, epochs),
        lr_scheduler_type="cosine",
        load_best_model_at_end=validation is not None,
        metric_for_best_model="f1" if validation is not None else None,
        greater_is_better=True,
        logging_steps=25,
        report_to="none",
        seed=seed,
        data_seed=seed,
        full_determinism=True,
        fp16=False,
        bf16=False,
        use_cpu=device.type == "cpu",
        optim="adamw_torch",
        gradient_accumulation_steps=accumulation,
        gradient_checkpointing=checkpointing,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        label_smoothing=float(config.get("label_smoothing", DEFAULT_LABEL_SMOOTHING)),
        model=model,
        args=args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if validation is not None else [],
    )

    trainer.train()
    return trainer, tokenizer


def select_epoch(
    config: dict[str, Any], training: pd.DataFrame, output_dir: Path, seed: int, selection_fold: int = 0
) -> dict[str, Any]:
    mask = build_inner_validation_mask(training, outer_fold=selection_fold)
    assignments = training[[c for c in ("Record_ID", "Template_Group", LABEL_COL) if c in training]].copy()
    assignments["role"] = np.where(mask, "epoch_validation", "epoch_train")
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_dir / "epoch_assignments.csv", index=False)
    trainer, tokenizer = train_phase(
        config, training.loc[~mask], training.loc[mask], output_dir / "selection", seed, int(config["epochs"])
    )
    history = [r for r in trainer.state.log_history if "eval_f1" in r]
    best = max(history, key=lambda row: (row["eval_f1"], -row["epoch"]))
    selection = {
        "selected_epoch": math.ceil(best["epoch"]),
        "inner_best_f1": best["eval_f1"],
        "epoch_train_rows": int((~mask).sum()),
        "epoch_validation_rows": int(mask.sum()),
        "seed": seed,
        "runtime": runtime_metadata(),
        "history": history,
    }
    atomic_json(output_dir / "epoch_selection.json", selection)
    del trainer, tokenizer
    release_memory()
    shutil.rmtree(output_dir / "selection", ignore_errors=True)
    return selection


def main(experiment_name: str = "") -> None:
    if not experiment_name:
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment_name", required=True)
        experiment_name = parser.parse_args().experiment_name
    config = yaml.safe_load((BASE_DIR / "params.yaml").read_text())
    experiment = next(e for e in config["experiments"] if e["name"] == experiment_name)
    bind_run()
    model_dir = SAVED_MODELS_DIR / experiment_name

    if artefact_matches_current_split(model_dir):
        return

    training = load_split("train")
    output = RESULTS_DIR / "training" / experiment_name
    fingerprint = identity({"split": current_split_manifest(), "config": experiment, "seed": RANDOM_STATE})

    with stage_status(output, fingerprint):
        selection = select_epoch(experiment, training, output, RANDOM_STATE)
        trainer, tokenizer = train_phase(
            experiment, training, None, output / "refit", RANDOM_STATE, selection["selected_epoch"]
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(model_dir)
        write_training_manifest(
            model_dir,
            experiment_name,
            {
                **experiment,
                **selection,
                "refit_training_rows": len(training),
                "loss_reduction": "weighted_sum_over_effective_batch_examples",
            },
        )
        del trainer, tokenizer
        release_memory()


if __name__ == "__main__":
    main()
