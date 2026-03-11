"""Fine-tuning transformer models for phishing detection.

Loads pre-trained transformer models, prepares tokenized datasets,
and trains them with weighted loss to handle class imbalance.
Saves trained models and metrics to disk and MLflow.
"""

import argparse
import os
from typing import Any, Dict, Tuple, Union

import mlflow
import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from src.config import BASE_DIR, LABEL_COL, SPLIT_DATA_DIR, TEXT_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_device() -> torch.device:
    """Get the optimal available device for training.

    Checks for availability of MPS (Apple Silicon) and CUDA (NVIDIA)
    devices. Defaults to CPU if no accelerator is available.

    Returns:
        torch.device: The best available device (mps, cuda, or cpu).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")

    elif torch.cuda.is_available():
        return torch.device("cuda")

    else:
        return torch.device("cpu")


def prepare_dataset(df: pd.DataFrame, tokenizer: Any, max_length: int) -> Dataset:
    """Prepare and tokenize a dataset from a DataFrame.

    Converts label columns to integers if necessary, ensures text columns
    are strings, converts the DataFrame to a Hugging Face Dataset, and
    applies tokenization.

    Args:
        df: Input pandas DataFrame containing text and labels.
        tokenizer: Hugging Face tokenizer instance to process text.
        max_length: Maximum sequence length for tokenization padding and
            truncation.

    Returns:
        Dataset: A Hugging Face Dataset object containing tokenized inputs
            and labels ready for training.
    """
    if df[LABEL_COL].dtype == bool:
        df[LABEL_COL] = df[LABEL_COL].astype(int)

    elif df[LABEL_COL].dtype == object:
        pass  # Assume already correct or handled elsewhere

    # Ensure text is string:
    df[TEXT_COL] = df[TEXT_COL].astype(str)

    dataset = Dataset.from_pandas(df[[TEXT_COL, LABEL_COL]])
    dataset = dataset.rename_column(LABEL_COL, "labels")

    def tokenize_function(examples: Dict[str, Any]) -> Any:
        return tokenizer(
            examples[TEXT_COL],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize_function, batched=True)


def load_and_prepare_data(tokenizer: Any, max_length: int) -> DatasetDict:
    """Load, process, and tokenize all dataset splits.

    Iterates through train, validation, and test splits stored as CSV
    files in ``SPLIT_DATA_DIR``, loading and preparing each using
    ``prepare_dataset``.

    Args:
        tokenizer: Hugging Face tokenizer instance used for processing.
        max_length: Maximum sequence length for tokenization.

    Returns:
        DatasetDict: A dictionary-like object containing 'train', 'val',
            and 'test' datasets.

    Raises:
        FileNotFoundError: If any required split file is missing.
    """
    data = {}

    for split in ["train", "val", "test"]:
        path = SPLIT_DATA_DIR / f"{split}.csv"

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path)
        data[split] = prepare_dataset(df, tokenizer, max_length)

    return DatasetDict(data)


def compute_metrics(pred: Any) -> Dict[str, Any]:
    """Compute evaluation metrics for the trainer.

    Calculates accuracy, precision, recall, and F1-score for binary
    classification tasks.

    Args:
        pred: A named tuple or object containing 'label_ids' and
            'predictions' (logits).

    Returns:
        dict: A dictionary mapping metric names (str) to their computed
            values (float).
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


class WeightedTrainer(Trainer):
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Union[torch.Tensor, int, None] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:

        if not isinstance(model, PreTrainedModel):
            raise TypeError("Expected PreTrainedModel")

        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Labels are missing from inputs")

        outputs = model(**inputs)
        logits = outputs["logits"]

        num_labels = model.config.num_labels

        weights = torch.tensor(
            [1.0, 5.0],
            device=logits.device,
            dtype=logits.dtype,
        )

        loss_fct = nn.CrossEntropyLoss(weight=weights)

        loss = loss_fct(
            logits.view(-1, num_labels),
            labels.view(-1).long(),
        )

        return (loss, outputs) if return_outputs else loss


# Main:
def main() -> None:
    """Fine-tune a transformer model for phishing detection.

    Loads experiment configuration from params.yaml, prepares tokenized
    datasets, initializes a transformer model, and trains it with weighted
    loss on imbalanced data. Saves the trained model and logs metrics to MLflow.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment to run from params.yaml",
    )
    args = parser.parse_args()

    params_path = BASE_DIR / "params.yaml"

    with open(params_path, "r") as f:
        full_config = yaml.safe_load(f)

    experiment_config = next(
        (exp for exp in full_config["experiments"] if exp["name"] == args.experiment_name),
        None,
    )

    if experiment_config is None:
        raise ValueError(f"Experiment '{args.experiment_name}' not found in params.yaml")

    model_name = experiment_config["model_name"]
    max_length = experiment_config["max_length"]
    batch_size = experiment_config["batch_size"]
    epochs = experiment_config["epochs"]
    learning_rate = float(experiment_config["learning_rate"])

    mlflow.set_experiment("phishing_transformer_finetune")
    mlflow.start_run(run_name=args.experiment_name)
    mlflow.log_params(experiment_config)

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading and preparing data...")
    tokenized_datasets = load_and_prepare_data(tokenizer, max_length)

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    device = get_device()
    logger.info(f"Using device: {device}")
    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    output_dir = f"./results/{args.experiment_name}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        logging_dir=f"./logs/{args.experiment_name}",
        report_to="mlflow",
    )

    # Use WeightedTrainer instead of standard Trainer:
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info(f"Starting weighted training for: {args.experiment_name}...")
    trainer.train()

    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")

    logger.info(f"Test results: {test_results}")
    mlflow.log_metrics(test_results)

    model_save_path = f"./saved_models/{args.experiment_name}"

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    logger.info(f"Saving model to {model_save_path}")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    mlflow.end_run()


# Entry point:
if __name__ == "__main__":
    main()
