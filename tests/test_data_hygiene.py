import unittest

import pandas as pd

from src.data.hygiene import (
    add_template_groups,
    assert_group_label_consistency,
    normalize_template,
    remove_generator_url_wrappers,
)
from src.data.preprocess_data import DataPreprocessor


class DataHygieneTests(unittest.TestCase):
    def test_markdown_google_wrapper_is_removed(self) -> None:
        text = "Kliknij [bank-secure.xyz/login](https://www.google.com/search?q=https://bank-secure.xyz/login)"
        cleaned = remove_generator_url_wrappers(text)
        self.assertEqual(cleaned, "Kliknij https://bank-secure.xyz/login")
        self.assertNotIn("google.com", cleaned)

    def test_direct_google_wrapper_is_unwrapped(self) -> None:
        text = "Status: https://www.google.com/search?q=dpd.com.pl"
        self.assertEqual(remove_generator_url_wrappers(text), "Status: dpd.com.pl")

    def test_variable_entities_share_one_template(self) -> None:
        left = "Dopłać 14.50 PLN: https://parcel-one.xyz/a/123"
        right = "Dopłać 29.99 PLN: https://parcel-two.xyz/b/987"
        self.assertEqual(normalize_template(left), normalize_template(right))

    def test_conflicting_group_labels_are_rejected(self) -> None:
        frame = add_template_groups(pd.DataFrame({"Content": ["Kod 123", "Kod 123"], "Is_Phishing": [False, True]}))
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            assert_group_label_consistency(frame)

    def test_model_text_excludes_generator_sender_annotation(self) -> None:
        record = {
            "Type": "E-mail",
            "Title": "Pilna wiadomość",
            "Sender_brand": "Inny",
            "Content": "Treść",
        }
        text = DataPreprocessor().build_text_field(record)
        self.assertIn("[TITLE] Pilna wiadomość", text)
        self.assertNotIn("[SENDER]", text)
        self.assertNotIn("Inny", text)

    def test_email_title_is_preserved_but_missing_sms_title_is_not_invented(self) -> None:
        preprocessor = DataPreprocessor()
        email = preprocessor.build_text_field({"Type": "E-mail", "Title": "Reset hasła", "Content": "Treść"})
        sms = preprocessor.build_text_field({"Type": "SMS", "Title": "Brak", "Content": "Treść"})
        self.assertIn("[TITLE] Reset hasła", email)
        self.assertNotIn("[TITLE]", sms)


if __name__ == "__main__":
    unittest.main()
