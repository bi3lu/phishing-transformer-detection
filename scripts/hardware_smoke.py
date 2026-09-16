"""Short offline MPS/CPU training probe; does not evaluate research examples."""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from src.config import BASE_DIR
from src.models.fine_tune import WeightedTrainer
from src.models.runtime import configure_runtime, release_memory, runtime_metadata
from src.models.training_schedule import resolve_training_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/phishing-hardware-smoke.json"))
    parser.add_argument("--batch-size", type=int, choices=[4, 8, 16])
    parser.add_argument("--updates", type=int, default=3)
    args = parser.parse_args()
    config = yaml.safe_load((BASE_DIR / "params.yaml").read_text())
    rows = []

    for experiment in config["experiments"]:
        device = configure_runtime()
        batch, eval_batch, accumulation, _ = resolve_training_schedule(experiment)

        if args.batch_size:
            batch, accumulation = args.batch_size, 16 // args.batch_size

        tokenizer = AutoTokenizer.from_pretrained(
            experiment["model_name"], revision=experiment["revision"], local_files_only=True
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            experiment["model_name"],
            revision=experiment["revision"],
            num_labels=2,
            attn_implementation="eager",
            local_files_only=True,
        )

        text = "[TYPE] SMS [CONTENT] To jest neutralna wiadomość do pomiaru wydajności. " * 100
        encoded = tokenizer([text] * batch, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        encoded["labels"] = torch.arange(batch) % 2

        with tempfile.TemporaryDirectory() as directory:
            trainer = WeightedTrainer(
                model=model,
                class_weights=torch.ones(2),
                label_smoothing=0.1,
                args=TrainingArguments(
                    output_dir=directory,
                    report_to="none",
                    use_cpu=device.type == "cpu",
                    per_device_train_batch_size=batch,
                    gradient_accumulation_steps=accumulation,
                    optim="adamw_torch",
                ),
            )
            trainer.current_gradient_accumulation_steps = accumulation
            trainer.create_optimizer()
            assert trainer.optimizer is not None

            timings = []

            for update in range(args.updates):
                start = time.perf_counter()
                loss = 0.0

                for _ in range(accumulation):
                    loss += float(trainer.training_step(model, dict(encoded), num_items_in_batch=batch * accumulation))

                trainer.optimizer.step()
                trainer.optimizer.zero_grad()

                if device.type == "mps":
                    torch.mps.synchronize()

                timings.append(time.perf_counter() - start)

            elapsed = timings[-1]
            rows.append(
                {
                    "model": experiment["name"],
                    **runtime_metadata(),
                    "batch": batch,
                    "accumulation": accumulation,
                    "sequence_length": 256,
                    "update_seconds": timings,
                    "last_update_seconds": elapsed,
                    "loss": loss,
                    "finite": bool(torch.isfinite(torch.tensor(loss))),
                    "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory() if device.type == "mps" else None,
                }
            )

            del trainer, model, tokenizer, encoded
            release_memory()

        print(json.dumps(rows[-1]), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
