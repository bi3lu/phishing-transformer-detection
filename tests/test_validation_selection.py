import unittest

import numpy as np

from src.evaluation.ensemble import select_ensemble
from src.evaluation.threshold_analysis import calculate_threshold_metrics, select_f1_threshold


class ValidationSelectionTests(unittest.TestCase):
    def test_threshold_is_selected_from_supplied_validation_predictions(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])
        selected = select_f1_threshold(calculate_threshold_metrics(labels, probabilities))
        self.assertGreater(float(selected["f1"]), 0.99)
        self.assertGreater(float(selected["threshold"]), 0.4)

    def test_ensemble_configuration_records_validation_provenance(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        predictions = {
            "a": np.asarray([0.1, 0.2, 0.8, 0.9]),
            "b": np.asarray([0.2, 0.3, 0.7, 0.8]),
            "c": np.asarray([0.4, 0.6, 0.4, 0.6]),
        }
        config, table = select_ensemble(predictions, labels)
        self.assertEqual(config["selection_split"], "validation")
        self.assertEqual(config["weighting"], "equal")
        self.assertFalse(table.empty)


if __name__ == "__main__":
    unittest.main()
