"""Feature extraction for phishing detection.

Extracts linguistic, structural, and security-related features from
email text to identify phishing patterns.
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List


class PhishingFeatureExtractor:
    """Extracts phishing-indicative features from email text.

    Identifies urgency keywords, threats, verification requests, suspicious
    URLs, homograph attacks, and suspicious domains.
    """

    def __init__(self) -> None:
        # TODO: Extract these lists to separate config file!
        self.urgency_keywords = [
            "natychmiast",
            "pilnie",
            "bez opóźnienia",
            "wygasa",
            "teraz",
            "szybko",
            "nie czekaj",
            "ostatnia szansa",
            "pośpiesz się",
        ]

        self.threat_keywords = [
            "zablokowane",
            "zablokuje",
            "wznowić",
            "przywrócić",
            "odblokować",
            "zmień hasło",
            "weryfikacja",
            "dezaktywacja",
            "zagrożenie",
            "niebezpieczeństwo",
        ]

        self.verification_keywords = ["potwierdź", "kod", "pin", "hasło", "tożsamość", "zaloguj", "login"]

        self.legit_domains = {
            "mbank.pl",
            "pkobp.pl",
            "ing.pl",
            "santander.pl",
            "aliorbank.pl",
            "poczta-polska.pl",
            "inpost.pl",
            "allegro.pl",
            "facebook.com",
        }

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text.

        Args:
            text: Text to extract URLs from.

        Returns:
            List of URLs found in the text.
        """
        return re.findall(r'https?://[^\s)"\']+', text.lower())

    def _get_domain(self, url: str) -> str:
        """Extract domain name from URL.

        Args:
            url: URL to extract domain from.

        Returns:
            Domain name without protocol.
        """
        domain = re.sub(r"https?://", "", url)
        return domain.split("/")[0].split(":")[0].split("@")[-1]

    def _check_homograph(self, domain: str) -> bool:
        """Check if a domain uses homograph attack techniques.

        Detects international characters and domain names similar to
        legitimate banking/service domains.

        Args:
            domain: Domain name to analyze.

        Returns:
            True if domain exhibits homograph attack characteristics.
        """
        # 1. Check for non-ASCII characters that might be lookalikes
        try:
            domain.encode("ascii")
        except UnicodeEncodeError:
            return True

        # 2. Check for similarity to known legitimate domains
        for legit in self.legit_domains:
            legit_name = legit.split(".")[0]
            domain_name = domain.split(".")[0]

            if 0.8 <= SequenceMatcher(None, legit_name, domain_name).ratio() < 1.0:
                return True

        return False

    def _count_triggers(self, text: str, word_list: List[str]) -> int:
        """Count occurrences of trigger words (with cap at 5).

        Args:
            text: Text to search in.
            word_list: List of keywords to count.

        Returns:
            Count of keyword occurrences, capped at 5.
        """
        count = 0
        text_lower = text.lower()

        for word in word_list:
            pattern = rf"\b{re.escape(word)}[a-z]*\b"
            count += len(re.findall(pattern, text_lower))

        return min(count, 5)

    def _normalize_for_stats(self, text: str) -> str:
        """Normalize text by replacing similar-looking characters.

        Substitutes lookalike characters (Cyrillic, digits) with ASCII
        equivalents to detect obfuscation attempts.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text with substitutions applied.
        """
        replacements = {"а": "a", "о": "o", "е": "e", "р": "p", "0": "o", "5": "s", "@": "a"}

        for char, rep in replacements.items():
            text = text.replace(char, rep)

        return text.lower()

    def get_all_features(self, text: str) -> Dict[str, int]:
        """Extract all phishing-related features from text.

        Computes urgency score, threat score, verification requests,
        URL counts, and suspicious domain indicators.

        Args:
            text: Email text to analyze.

        Returns:
            Dictionary mapping feature names to their extracted values.
        """
        text = self._normalize_for_stats(text)
        urls = self._extract_urls(text)

        features = {
            "urgency_score": self._count_triggers(text, self.urgency_keywords),
            "threat_score": self._count_triggers(text, self.threat_keywords),
            "verification_request": self._count_triggers(text, self.verification_keywords),
            "num_urls": len(urls),
            "has_suspicious_tld": 0,
            "has_homograph_attack": 0,
            "has_url_shortener": 0,
        }

        suspicious_tlds = [".net", ".info", ".xyz", ".tk", ".ga", ".cf", ".gq"]
        shorteners = ["bit.ly", "tinyurl.com", "t.co", "is.gd"]

        for url in urls:
            domain = self._get_domain(url)

            if any(tld in domain for tld in suspicious_tlds):
                features["has_suspicious_tld"] = 1

            if any(short in domain for short in shorteners):
                features["has_url_shortener"] = 1

            if self._check_homograph(domain):
                features["has_homograph_attack"] = 1

        return features
