"""Regression cases discovered by the repository audit."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments

from src.data.cv_folds import build_cv_folds, outer_training_mask
from src.data.hygiene import add_template_groups, assert_group_label_consistency, normalize_template
from src.data.schema import validate_records
from src.data.template_audit import cluster_templates
from src.evaluation.threshold_analysis import calculate_threshold_metrics, select_f1_threshold
from src.evaluation.uncertainty import (
    binary_classification_metrics,
    grouped_bootstrap_confidence_intervals,
    template_group_binomial_confidence_intervals,
)
from src.features.extractor import PhishingFeatureExtractor
from src.models.fine_tune import WeightedTrainer
from src.models.provenance import current_split_manifest
from src.models.runtime import configure_runtime
from src.utils.artifacts import file_sha256, stage_status


class DataRegressionTests(unittest.TestCase):
    def test_dates_identifiers_and_new_tlds(self) -> None:
        pairs = [
            ("Od 1 marca 2025 zmiany 65 PLN", "Od 5 kwietnia 2025 zmiany 70 PLN"),
            ("Numer ZW8K9P2Q7R4T", "Numer AB54R"),
            ("Kliknij bank.site/a", "Kliknij bank.biz/b"),
        ]
        for left, right in pairs:
            self.assertEqual(normalize_template(left), normalize_template(right))

    def test_clustering_preserves_labels_and_catches_near_variants(self) -> None:
        df = pd.DataFrame(
            {
                "Content": [
                    "Kod autoryzacyjny Santander: 123 456. Transakcja: Zmiana limitow. Nie podawaj kodu nikomu.",
                    "Kod autoryzacyjny Santander: 928 103. Transakcja: Zmiana limitow karty. Nie podawaj kodu nikomu.",
                ],
                "Is_Phishing": [0, 1],
            }
        )
        clustered, audit = cluster_templates(df)
        self.assertEqual(clustered.Template_Group.nunique(), 1)
        self.assertEqual(clustered.Is_Phishing.tolist(), [0, 1])
        self.assertFalse(audit.empty)
        assert_group_label_consistency(clustered)

    def test_schema_rejects_invalid_label_and_empty_content(self) -> None:
        df = pd.DataFrame(
            {
                "Type": ["SMS"],
                "Title": ["Brak"],
                "Content": ["test"],
                "Text": ["test"],
                "Model_Source": ["x"],
                "Is_Phishing": [2],
            }
        )
        with self.assertRaises(ValueError):
            validate_records(df)
        df["Is_Phishing"] = 0
        df["Content"] = ""
        with self.assertRaises(ValueError):
            validate_records(df)

    def test_source_holdout_purges_shared_templates(self) -> None:
        df = pd.DataFrame(
            {
                "Text": ["a", "b", "c", "d", "e", "f"],
                "Is_Phishing": [0, 1, 0, 1, 0, 1],
                "Model_Source": ["a", "a", "b", "b", "c", "c"],
                "Template_Group": ["shared", "x", "shared", "y", "z", "w"],
            }
        )
        folds = build_cv_folds(df, 3, "source_holdout")
        held = int(folds.iloc[0])
        mask = outer_training_mask(df, folds, held)
        self.assertFalse(mask.iloc[2])
        self.assertFalse(set(df.loc[mask, "Template_Group"]) & set(df.loc[folds == held, "Template_Group"]))

    def test_features_ignore_tags_and_do_not_call_all_idn_attacks(self) -> None:
        extractor = PhishingFeatureExtractor()
        self.assertEqual(extractor.get_all_features("[TYPE] SMS [CONTENT] Spokojna wiadomość")["emo_score"], 0)
        features = extractor.get_all_features("https://żółw.pl")
        self.assertEqual(features["has_idn_domain"], 1)
        self.assertEqual(features["has_homograph_attack"], 0)


class TrainingRegressionTests(unittest.TestCase):
    def test_accumulated_gradients_match_full_batch_with_weights_and_partial_batch(self) -> None:
        config_class: Any = BertConfig
        model_class: Any = BertForSequenceClassification
        config = config_class(
            vocab_size=20,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
        configure_runtime(42)
        model = model_class(config)
        with tempfile.TemporaryDirectory() as directory:
            trainer = WeightedTrainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory, use_cpu=True, report_to="none", gradient_accumulation_steps=4
                ),
                class_weights=torch.tensor([2.0, 0.65]),
                label_smoothing=0.1,
            )
            inputs = torch.randint(0, 20, (14, 5))
            labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1])
            trainer.current_gradient_accumulation_steps = 1
            trainer.training_step(model, {"input_ids": inputs, "labels": labels}, num_items_in_batch=torch.tensor(14))
            full = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            model.zero_grad()
            trainer.current_gradient_accumulation_steps = 4
            for start in range(0, 14, 4):
                trainer.training_step(
                    model,
                    {"input_ids": inputs[start : start + 4], "labels": labels[start : start + 4]},
                    num_items_in_batch=torch.tensor(14),
                )
            actual = [p.grad for p in model.parameters() if p.grad is not None]
            for left, right in zip(full, actual):
                torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-4)

    def test_seed_is_applied_before_random_head(self) -> None:
        config_class: Any = BertConfig
        model_class: Any = BertForSequenceClassification
        config = config_class(
            vocab_size=20, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16
        )
        configure_runtime(42)
        first = model_class(config).classifier.weight.detach().clone()
        torch.rand(100)
        configure_runtime(42)
        second = model_class(config).classifier.weight.detach()
        torch.testing.assert_close(first, second)


class IntegrityAndStatisticsTests(unittest.TestCase):
    def test_split_guard_reads_physical_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("canonical")
            train = root / "train.csv"
            train.write_text("original")
            from src.config import PROTOCOL_VERSION

            manifest = {
                "protocol_version": PROTOCOL_VERSION,
                "source_dataset": str(source),
                "source_sha256": file_sha256(source),
                "splits": {"train": {"sha256": file_sha256(train)}},
            }
            (root / "split_manifest.json").write_text(json.dumps(manifest))
            with patch("src.models.provenance.SPLIT_DATA_DIR", root):
                current_split_manifest(("train",))
                train.write_text("tampered")
                with self.assertRaisesRegex(ValueError, "Physical"):
                    current_split_manifest(("train",))

    def test_failed_stage_cannot_look_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                with stage_status(root, "fixed"):
                    raise RuntimeError("interrupted")
            self.assertEqual(json.loads((root / "status.json").read_text())["status"], "failed")
            with self.assertRaises(ValueError):
                with stage_status(root, "changed"):
                    pass

    def test_exact_threshold_search_includes_narrow_optimum(self) -> None:
        row = select_f1_threshold(calculate_threshold_metrics(np.array([0, 1]), np.array([0.501, 0.509])))
        self.assertEqual(row.f1, 1.0)

    def test_optimized_bootstrap_matches_reference_with_ties_and_mixed_groups(self) -> None:
        y = np.array([0, 1, 0, 1, 1, 0])
        p = np.array([0.1, 0.1, 0.4, 0.8, 0.8, 0.9])
        g = np.array(["a", "a", "b", "b", "c", "d"])
        actual = grouped_bootstrap_confidence_intervals(y, p, g, n_resamples=100)
        unique = np.unique(g)
        rng = np.random.default_rng(42)
        values: dict[str, list[float]] = {name: [] for name in actual}
        for _ in range(100):
            chosen = rng.integers(0, len(unique), size=len(unique))
            rows = np.concatenate([np.flatnonzero(g == unique[i]) for i in chosen])
            for name, value in binary_classification_metrics(y[rows], p[rows]).items():
                if np.isfinite(value):
                    values[name].append(value)
        for name, draws in values.items():
            self.assertAlmostEqual(actual[name]["ci_lower"], np.quantile(draws, 0.025))
            self.assertAlmostEqual(actual[name]["ci_upper"], np.quantile(draws, 0.975))
        grouped = template_group_binomial_confidence_intervals(y, p, g)
        self.assertIn("group_all_correct", grouped)


class NewDeliveryTests(unittest.TestCase):
    def test_joined_records_and_multiline_content_keep_origin(self) -> None:
        from src.data.preprocess_data import DataPreprocessor

        text = "ID:002|Type:Notification|Title:Brak|Content:Pierwsza\nwiadomość|Is_Phishing:False ID:003|Type:Email|Title:Temat|Content:Druga|Is_Phishing:True"
        rows = list(DataPreprocessor.iter_records(text))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][2]["Content"], "Pierwsza\nwiadomość")
        self.assertEqual(rows[1][0], 2)
        self.assertEqual(rows[1][2]["Source_Record_ID"], "003")
        self.assertEqual(rows[1][2]["Is_Phishing"], "True")

    def test_notebook_sources_have_valid_python(self) -> None:
        import ast

        from src.config import BASE_DIR

        for path in (BASE_DIR / "notebooks").glob("*.ipynb"):
            notebook = json.loads(path.read_text())
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()
