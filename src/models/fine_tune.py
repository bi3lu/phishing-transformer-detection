import os
from dataclasses import dataclass

import mlflow  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
from datasets import Dataset, DatasetDict  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification  # type: ignore
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import LABEL_COL, SPLIT_DATA_DIR, TEXT_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    MODEL_NAME: str = "allegro/herbert-base-cased"
    MAX_LENGTH: int = 512
    BATCH_SIZE: int = 8
    EPOCHS: int = 3
    LEARNING_RATE: float = 2e-5


def load_data():  # TODO: Add docstring
    data = {}

    for split in ["train", "val", "test"]:
        path = SPLIT_DATA_DIR / f"{split}.csv"

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path)

        # Ensure labels are integers
        if df[LABEL_COL].dtype == bool:
            df[LABEL_COL] = df[LABEL_COL].astype(int)

        elif df[LABEL_COL].dtype == object:
            pass

        # Ensure text is string
        df[TEXT_COL] = df[TEXT_COL].astype(str)

        dataset = Dataset.from_pandas(df[[TEXT_COL, LABEL_COL]])

        # Rename label column to 'labels' for Trainer
        dataset = dataset.rename_column(LABEL_COL, "labels")
        data[split] = dataset

    return DatasetDict(data)


def compute_metrics(pred):  # TODO: Add docstring
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():  # TODO: Add docstring
    mlflow.set_experiment("phishing_transformer_finetune")
    mlflow.start_run()

    # Log parameters
    mlflow.log_params(
        {
            "model_name": ModelConfig.MODEL_NAME,
            "max_length": ModelConfig.MAX_LENGTH,
            "batch_size": ModelConfig.BATCH_SIZE,
            "epochs": ModelConfig.EPOCHS,
            "learning_rate": ModelConfig.LEARNING_RATE,
        }
    )

    logger.info("Loading data...")
    dataset = load_data()

    logger.info(f"Loading tokenizer: {ModelConfig.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.MODEL_NAME)

    def _tokenize_function(examples):  # TODO: Add docstring
        return tokenizer(
            examples[TEXT_COL],
            padding="max_length",
            truncation=True,
            max_length=ModelConfig.MAX_LENGTH,
        )

    logger.info("Tokenizing data...")
    tokenized_datasets = dataset.map(_tokenize_function, batched=True)

    logger.info(f"Loading model: {ModelConfig.MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        ModelConfig.MODEL_NAME, num_labels=2
    )

    if torch.backends.mps.is_available():
        device = "mps"

    elif torch.cuda.is_available():
        device = "cuda"

    else:
        device = "cpu"

    logger.info(f"Using device: {device}")
    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=ModelConfig.LEARNING_RATE,
        per_device_train_batch_size=ModelConfig.BATCH_SIZE,
        per_device_eval_batch_size=ModelConfig.BATCH_SIZE,
        num_train_epochs=ModelConfig.EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        logging_dir="./logs",
        report_to="mlflow",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(
        tokenized_datasets["test"], metric_key_prefix="test"
    )
    logger.info(f"Test results: {test_results}")
    mlflow.log_metrics(test_results)

    model_save_path = "./saved_models/fine_tuned_bert"

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    logger.info(f"Saving model to {model_save_path}")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    mlflow.end_run()


if __name__ == "__main__":
    main()
