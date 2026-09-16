import unittest

import yaml

from src.config import BASE_DIR
from src.evaluation.inference import registered_model_names


class FinalModelScopeTests(unittest.TestCase):
    def test_final_scope_contains_exactly_three_transformers(self) -> None:
        self.assertEqual(
            registered_model_names(),
            ["herbert-base", "polish-roberta-v2", "distilbert-multilingual"],
        )

    def test_scope_amendment_is_recorded(self) -> None:
        with (BASE_DIR / "params.yaml").open("r", encoding="utf-8") as handle:
            protocol = yaml.safe_load(handle)["research_protocol"]
        amendment = protocol["model_scope_amendment"]
        self.assertEqual(amendment["date"], "2026-07-16")
        self.assertIn("resource", amendment["reason"].lower())


if __name__ == "__main__":
    unittest.main()
