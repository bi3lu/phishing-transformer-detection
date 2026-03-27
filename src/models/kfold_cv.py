"""K-Fold cross-validation for transformer phishing detection models.

Runs stratified K-Fold CV to produce more reliable performance estimates
on the small dataset (~2800 training samples).  Each fold trains a fresh
model, evaluates on the held-out split, and reports aggregated metrics.
"""

import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)

from src.config import BASE_DIR, LABEL_COL, SPLIT_DATA_DIR, TEXT_COL
from src.data.augmented_dataset import AugmentedPhishingDataset
from src.models.fine_tune import (
    DEFAULT_FREEZE_LAYERS,
    DEFAULT_LABEL_SMOOTHING,
    WeightedTrainer,
    compute_metrics,
    freeze_lower_layers,
    get_device,
    prepare_dataset,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_kfold(
    experiment_name: str,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Run stratified K-Fold cross-validation for a single experiment.

    Args:
        experiment_name: Name of the experiment from params.yaml.
        n_splits: Number of CV folds.

    Returns:
        DataFrame with per-fold and mean/std metrics.
    """
    params_path = BASE_DIR / "params.yaml"

    with open(params_path, "r") as f:
        full_config = yaml.safe_load(f)

    experiment_config = next(
        (exp for exp in full_config["experiments"] if exp["name"] == experiment_name),
        None,
    )

    if experiment_config is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in params.yaml")

    model_name = experiment_config["model_name"]
    max_length = experiment_config["max_length"]
    batch_size = experiment_config["batch_size"]
    epochs = experiment_config["epochs"]
    learning_rate = float(experiment_config["learning_rate"])

    # Load full training + validation data for CV:
    train_df = pd.read_csv(SPLIT_DATA_DIR / "train.csv")
    val_df = pd.read_csv(SPLIT_DATA_DIR / "val.csv")
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    texts = full_df[TEXT_COL].astype(str).values
    labels = full_df[LABEL_COL].map({False: 0, True: 1}).astype(int).values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = get_device()

    fold_results: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        logger.info(f"=== Fold {fold_idx + 1}/{n_splits} ===")

        # Prepare fold data:
        fold_train_texts = texts[train_idx].tolist()
        fold_train_labels = labels[train_idx].tolist()
        fold_val_df = full_df.iloc[val_idx].copy()

        # On-the-fly augmented training dataset:
        train_dataset = AugmentedPhishingDataset(
            texts=fold_train_texts,
            labels=fold_train_labels,
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            max_length=max_length,
            augment=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        val_dataset = prepare_dataset(fold_val_df, tokenizer, max_length)

        # Fresh model for each fold:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        freeze_layers = int(experiment_config.get("freeze_layers", DEFAULT_FREEZE_LAYERS))
        
        if freeze_layers > 0:
            freeze_lower_layers(model, num_layers_to_freeze=freeze_layers)

        model.to(device)

        # Class weights for this fold:
        fold_label_counts = pd.Series(fold_train_labels).value_counts().sort_index()
        fold_total = len(fold_train_labels)
        class_weights = torch.tensor([fold_total / (2 * fold_label_counts[c]) for c in sorted(fold_label_counts.index)])

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir=f"./results/kfold_{experiment_name}/fold_{fold_idx}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=50,
            report_to="none",
            seed=42 + fold_idx,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=4,
        )

        label_smoothing = float(experiment_config.get("label_smoothing", DEFAULT_LABEL_SMOOTHING))
        trainer = WeightedTrainer(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        eval_results = trainer.evaluate(val_dataset, metric_key_prefix="eval")

        fold_metrics = {
            "fold": fold_idx + 1,
            "f1": eval_results["eval_f1"],
            "precision": eval_results["eval_precision"],
            "recall": eval_results["eval_recall"],
            "accuracy": eval_results["eval_accuracy"],
            "roc_auc": eval_results["eval_roc_auc"],
        }
        fold_results.append(fold_metrics)

        logger.info(
            f"Fold {fold_idx + 1}: F1={fold_metrics['f1']:.4f} "
            f"Precision={fold_metrics['precision']:.4f} "
            f"Recall={fold_metrics['recall']:.4f} "
            f"AUC={fold_metrics['roc_auc']:.4f}"
        )

        # Cleanup to free memory:
        del model, trainer
        import gc

        gc.collect()
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate results:
    results_df = pd.DataFrame(fold_results)

    summary = {
        "fold": "MEAN ± STD",
        "f1": f"{results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}",
        "precision": f"{results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}",
        "recall": f"{results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}",
        "accuracy": f"{results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}",
        "roc_auc": f"{results_df['roc_auc'].mean():.4f} ± {results_df['roc_auc'].std():.4f}",
    }
    summary_df = pd.DataFrame([summary])
    final_df = pd.concat([results_df, summary_df], ignore_index=True)

    logger.info(f"\n{final_df.to_string(index=False)}")

    output_path = BASE_DIR / "results" / f"kfold_{experiment_name}.csv"
    final_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    return final_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run K-Fold CV for a transformer experiment.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Experiment name from params.yaml",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    args = parser.parse_args()

    run_kfold(experiment_name=args.experiment_name, n_splits=args.n_splits)


if __name__ == "__main__":
    main()
