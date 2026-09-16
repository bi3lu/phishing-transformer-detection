import unittest
from unittest.mock import Mock, patch

from src.data.augmented_dataset import AugmentedPhishingDataset


class AugmentationContractTests(unittest.TestCase):
    def test_augmentation_is_off_by_default_and_never_adds_feature_tags(self) -> None:
        tokenizer = Mock(return_value={"input_ids": [1], "attention_mask": [1]})
        dataset = AugmentedPhishingDataset(
            ["[TYPE] SMS\n[CONTENT] Wiadomość"],
            [1],
            tokenizer,
        )
        dataset[0]
        passed_text = tokenizer.call_args.args[0]
        self.assertEqual(passed_text, "[TYPE] SMS\n[CONTENT] Wiadomość")
        self.assertNotIn("[FEAT", passed_text)

    def test_both_labels_use_the_same_augmentation_path(self) -> None:
        tokenizer = Mock(return_value={"input_ids": [1], "attention_mask": [1]})
        dataset = AugmentedPhishingDataset(
            ["[CONTENT] legal", "[CONTENT] phish"],
            [0, 1],
            tokenizer,
            augment=True,
        )
        with (
            patch.object(dataset.augmenter, "introduce_typos", side_effect=lambda text: f"T({text})"),
            patch.object(dataset.augmenter, "apply_homoglyphs", side_effect=lambda text: f"H({text})"),
        ):
            dataset[0]
            first = tokenizer.call_args.args[0]
            dataset[1]
            second = tokenizer.call_args.args[0]

        self.assertEqual(first, "[CONTENT] H(T(legal))")
        self.assertEqual(second, "[CONTENT] H(T(phish))")
        self.assertNotIn("[FEAT", first + second)
        self.assertNotIn("[BRAND]", first + second)


if __name__ == "__main__":
    unittest.main()
