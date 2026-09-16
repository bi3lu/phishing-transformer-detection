import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation.analysis import mcnemar_test
from src.evaluation.calibration import calibration_metrics, cross_fitted_calibration
from src.evaluation.threshold_analysis import calculate_deployment_scenarios
from src.evaluation.uncertainty import (
    grouped_bootstrap_confidence_intervals,
    template_group_binomial_confidence_intervals,
)


class StatisticsAndCalibrationTests(unittest.TestCase):
    def test_calibration_metrics_include_brier_and_ece(self) -> None:
        labels = np.asarray([0, 1])
        probabilities = np.asarray([0.1, 0.9])
        metrics = calibration_metrics(labels, probabilities, n_bins=10)
        self.assertAlmostEqual(metrics["brier_score"], 0.01)
        self.assertAlmostEqual(metrics["ece"], 0.1)

    def test_calibration_cross_fitting_keeps_groups_together(self) -> None:
        labels = np.asarray([(index // 2) % 2 for index in range(100)])
        probabilities = np.linspace(0.05, 0.95, 100)
        groups = np.asarray([f"group_{index // 2}" for index in range(100)])
        calibrated, _, folds = cross_fitted_calibration(probabilities, labels, groups, n_splits=5)
        assignments = pd.DataFrame({"group": groups, "fold": folds})
        self.assertFalse(np.isnan(calibrated).any())
        self.assertEqual(assignments.groupby("group")["fold"].nunique().max(), 1)

    def test_deployment_scenarios_use_assumed_prevalence(self) -> None:
        threshold_metrics = pd.DataFrame([{"threshold": 0.5, "tpr": 0.8, "fpr": 0.1, "fnr": 0.2}])
        scenarios = [{"name": "one_percent", "prevalence": 0.01, "fp_cost": 1.0, "fn_cost": 1.0}]
        row = calculate_deployment_scenarios(threshold_metrics, scenarios).iloc[0]
        expected_precision = 0.01 * 0.8 / (0.01 * 0.8 + 0.99 * 0.1)
        self.assertAlmostEqual(float(row["deployment_precision"]), expected_precision)
        self.assertAlmostEqual(float(row["expected_cost_per_message"]), 0.101)

    def test_mcnemar_is_exact_group_level_and_holm_corrected(self) -> None:
        group_labels = np.asarray([index % 2 for index in range(20)])
        correct = np.where(group_labels == 1, 0.9, 0.1)
        model_b = correct.copy()
        model_b[:10] = 1.0 - model_b[:10]
        model_c = 1.0 - correct

        labels = np.repeat(group_labels, 2)
        groups = np.repeat(np.asarray([f"template_{index}" for index in range(20)]), 2)
        predictions = {
            "a": np.repeat(correct, 2),
            "b": np.repeat(model_b, 2),
            "c": np.repeat(model_c, 2),
        }
        with tempfile.TemporaryDirectory() as directory, patch("src.evaluation.analysis.RESULTS_DIR", Path(directory)):
            results = mcnemar_test(predictions, labels, groups)

        self.assertEqual(len(results), 3)
        self.assertTrue((results["correction"] == "Holm").all())
        self.assertTrue(results["reject_holm"].all())
        self.assertLessEqual(int(results["discordant_template_groups"].max()), 20)

    def test_bootstrap_resamples_whole_template_groups(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 1])
        probabilities = np.asarray([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        groups = np.asarray(["a", "a", "b", "b", "c", "d"])
        intervals = grouped_bootstrap_confidence_intervals(
            labels,
            probabilities,
            groups,
            n_resamples=200,
            random_state=42,
        )
        self.assertEqual(intervals["f1"]["template_groups"], 4)
        self.assertEqual(intervals["f1"]["successful_resamples"], 200)
        self.assertAlmostEqual(intervals["f1"]["estimate"], 1.0)
        self.assertLessEqual(intervals["f1"]["ci_lower"], intervals["f1"]["ci_upper"])

    def test_exact_group_interval_is_not_degenerate_for_zero_errors(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
        groups = np.asarray(["a", "b", "c", "d"])
        intervals = template_group_binomial_confidence_intervals(labels, probabilities, groups)
        accuracy = intervals["group_all_correct"]
        self.assertEqual(accuracy["estimate"], 1.0)
        self.assertLess(accuracy["ci_lower"], 1.0)
        self.assertEqual(accuracy["ci_upper"], 1.0)


if __name__ == "__main__":
    unittest.main()
