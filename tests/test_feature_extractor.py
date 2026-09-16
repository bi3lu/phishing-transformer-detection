import unittest

from src.features.extractor import PhishingFeatureExtractor


class FeatureExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = PhishingFeatureExtractor()

    def test_decimal_amount_is_detected_before_keyword_normalization(self) -> None:
        features = self.extractor.get_all_features("Do zapłaty: 14.50 PLN")
        self.assertEqual(features["fin_score"], 1)

    def test_unicode_and_punycode_homographs_are_detected(self) -> None:
        unicode_features = self.extractor.get_all_features("Wejdź na https://раypal.com/login")
        punycode_features = self.extractor.get_all_features("Wejdź na xn--pypal-4ve.com/login")
        self.assertEqual(unicode_features["has_homograph_attack"], 1)
        self.assertEqual(punycode_features["has_homograph_attack"], 1)

    def test_bare_domain_is_counted_as_url(self) -> None:
        features = self.extractor.get_all_features("Status przesyłki: inpost-paczka.pl/status/123")
        self.assertEqual(features["num_urls"], 1)

    def test_google_wrapper_exposes_target_domain(self) -> None:
        text = "Kliknij https://www.google.com/search?q=https%3A%2F%2Fbank-alert.xyz%2Flogin"
        features = self.extractor.get_all_features(text)
        self.assertEqual(features["num_urls"], 1)
        self.assertEqual(features["has_suspicious_tld"], 1)

    def test_suspicious_tld_is_an_exact_suffix_not_a_substring(self) -> None:
        safe = self.extractor.get_all_features("https://portal.xyz.example.pl/login")
        suspicious = self.extractor.get_all_features("https://portal.example.xyz/login")
        self.assertEqual(safe["has_suspicious_tld"], 0)
        self.assertEqual(suspicious["has_suspicious_tld"], 1)

    def test_shortener_requires_a_matching_host(self) -> None:
        safe = self.extractor.get_all_features("https://not-bit.ly.example.com/path")
        short = self.extractor.get_all_features("https://bit.ly/path")
        self.assertEqual(safe["has_url_shortener"], 0)
        self.assertEqual(short["has_url_shortener"], 1)


if __name__ == "__main__":
    unittest.main()
