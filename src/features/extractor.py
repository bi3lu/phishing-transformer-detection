"""Feature extraction for phishing detection.

Extracts linguistic, structural, and security-related features from
email text to identify phishing patterns.
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List

from src.features.extractor_config import (
    ACTION_KEYWORDS,
    FINANCIAL_KEYWORDS,
    LEGIT_DOMAINS,
    THREAT_KEYWORD,
    URGENCY_KEYWORD,
    VERIFICATION_KEYWORD,
)


class PhishingFeatureExtractor:
    """Extracts phishing-indicative features from email text.

    Identifies urgency keywords, threats, verification requests, suspicious
    URLs, homograph attacks, and suspicious domains.
    """

    def __init__(self) -> None:
        pass

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
        for legit in LEGIT_DOMAINS:
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

    def _calculate_emotionality(self, text: str) -> int:
        exc_count = text.count("!")
        ques_count = text.count("?")
        caps_words = len(re.findall(r"\b[A-Z]{3,}\b", text))
        score = exc_count + ques_count + caps_words

        return min(score, 5)

    def _get_financial_index(self, text: str, words: List[str]) -> int:
        keyword_count = self._count_triggers(text, words)
        money_patterns = len(re.findall(r"\d+[\.,]\d{2}\s?(?:PLN|zł|eur|usd|€|\$)", text, re.IGNORECASE))

        return min(keyword_count + money_patterns, 5)

    def get_all_features(self, text: str) -> Dict[str, int]:
        """Extract all phishing-related features from text.

        Computes urgency score, threat score, verification requests,
        URL counts, and suspicious domain indicators.

        Args:
            text: Email text to analyze.

        Returns:
            Dictionary mapping feature names to their extracted values.
        """
        raw_text = text
        normalized_text = self._normalize_for_stats(text).lower()
        urls = self._extract_urls(normalized_text)

        features = {
            "urgency_score": self._count_triggers(normalized_text, URGENCY_KEYWORD),
            "threat_score": self._count_triggers(normalized_text, THREAT_KEYWORD),
            "verif_score": self._count_triggers(normalized_text, VERIFICATION_KEYWORD),
            "action_score": self._count_triggers(normalized_text, ACTION_KEYWORDS),
            "fin_score": self._get_financial_index(normalized_text, FINANCIAL_KEYWORDS),
            "emo_score": self._calculate_emotionality(raw_text),
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
