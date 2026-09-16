"""Auditable heuristic features for phishing-message analysis.

The transformer input does not contain these features.  They are retained for
descriptive analysis and therefore must be calculated from observable message
text without destroying evidence before it is inspected.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, List
from urllib.parse import urlsplit

from src.data.hygiene import remove_generator_url_wrappers
from src.features.extractor_config import (
    ACTION_KEYWORDS,
    FINANCIAL_KEYWORDS,
    LEGIT_DOMAINS,
    THREAT_KEYWORD,
    URGENCY_KEYWORD,
    VERIFICATION_KEYWORD,
)

SCHEME_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()\[\]{}\"']+", re.IGNORECASE)
BARE_DOMAIN_RE = re.compile(
    r"(?<![@\w.-])"
    r"(?:[a-z0-9\u0080-\uffff](?:[a-z0-9\u0080-\uffff-]{0,62})\.)+"
    r"(?:[a-z\u0080-\uffff]{2,63}|xn--[a-z0-9-]{2,59})"
    r"(?::\d{1,5})?(?:/[^\s<>()\[\]{}\"']*)?",
    re.IGNORECASE,
)
MONEY_RE = re.compile(
    r"(?<!\w)(?:\d{1,3}(?:[ .]\d{3})+|\d+)(?:[.,]\d{2})?\s*" r"(?:PLN|zł|złotych|EUR|USD|€|\$)(?!\w)",
    re.IGNORECASE,
)
TRAILING_URL_PUNCTUATION = ".,;:!?)]}>\"'"
SUSPICIOUS_TLDS = frozenset({"net", "info", "xyz", "tk", "ga", "cf", "gq"})
URL_SHORTENERS = frozenset({"bit.ly", "tinyurl.com", "t.co", "is.gd"})
FEATURE_NAMES = (
    "urgency_score",
    "threat_score",
    "verif_score",
    "action_score",
    "fin_score",
    "emo_score",
    "num_urls",
    "has_suspicious_tld",
    "has_homograph_attack",
    "has_idn_domain",
    "has_url_shortener",
)


class PhishingFeatureExtractor:
    """Extract reproducible linguistic and URL-security features."""

    def _extract_urls(self, text: str) -> List[str]:
        """Extract scheme URLs, ``www`` URLs, and bare domains once each."""
        cleaned = remove_generator_url_wrappers(str(text))
        candidates: List[str] = []
        occupied = [False] * len(cleaned)

        for match in SCHEME_URL_RE.finditer(cleaned):
            candidates.append(match.group(0).rstrip(TRAILING_URL_PUNCTUATION))

            for index in range(match.start(), match.end()):
                occupied[index] = True

        for match in BARE_DOMAIN_RE.finditer(cleaned):
            if any(occupied[match.start() : match.end()]):
                continue

            candidates.append(match.group(0).rstrip(TRAILING_URL_PUNCTUATION))

        # Stable de-duplication is important for reproducible feature counts:
        return list(dict.fromkeys(candidate for candidate in candidates if candidate))

    def _get_domain(self, url: str) -> str:
        """Return the parsed hostname for a scheme URL or bare domain."""
        candidate = str(url).strip().rstrip(TRAILING_URL_PUNCTUATION)
        parsed = urlsplit(candidate if re.match(r"^[a-z][a-z0-9+.-]*://", candidate, re.I) else f"//{candidate}")
        return (parsed.hostname or "").casefold().rstrip(".")

    @staticmethod
    def _domain_label(domain: str) -> str:
        labels = domain.rstrip(".").split(".")
        return labels[-2] if len(labels) >= 2 else labels[0]

    @staticmethod
    def _homograph_skeleton(value: str) -> str:
        replacements = str.maketrans(
            {
                "а": "a",
                "е": "e",
                "о": "o",
                "р": "p",
                "с": "c",
                "у": "y",
                "х": "x",
                "і": "i",
                "ј": "j",
                "0": "o",
                "1": "l",
                "3": "e",
                "5": "s",
            }
        )
        return unicodedata.normalize("NFKC", value).casefold().translate(replacements)

    def _check_homograph(self, domain: str) -> bool:
        """Detect Unicode/punycode hosts and lookalikes of known domains."""
        if not domain:
            return False

        try:
            domain = domain.encode("ascii").decode("idna")

        except (UnicodeError, ValueError):
            pass

        domain_label = self._domain_label(domain)
        domain_skeleton = self._homograph_skeleton(domain_label)

        for legit in LEGIT_DOMAINS:
            legit_label = self._domain_label(legit.casefold())

            if domain_label != legit_label and domain_skeleton == self._homograph_skeleton(legit_label):
                return True

            similarity = SequenceMatcher(None, legit_label, domain_label).ratio()

            if 0.8 <= similarity < 1.0:
                return True

        return False

    def _count_triggers(self, text: str, word_list: List[str]) -> int:
        """Count keyword occurrences, capped at five."""
        count = 0
        text_lower = text.casefold()

        for word in word_list:
            pattern = rf"\b{re.escape(word.casefold())}[a-ząćęłńóśźż]*\b"
            count += len(re.findall(pattern, text_lower))

        return min(count, 5)

    def _normalize_for_keywords(self, text: str) -> str:
        """Normalize lookalike letters only after raw security evidence is read."""
        return self._homograph_skeleton(unicodedata.normalize("NFKC", text))

    def _calculate_emotionality(self, text: str) -> int:
        exc_count = text.count("!")
        ques_count = text.count("?")
        caps_words = len(re.findall(r"\b[A-ZĄĆĘŁŃÓŚŹŻ]{3,}\b", text))
        return min(exc_count + ques_count + caps_words, 5)

    def _get_financial_index(self, keyword_text: str, raw_text: str, words: List[str]) -> int:
        keyword_count = self._count_triggers(keyword_text, words)
        money_patterns = len(MONEY_RE.findall(unicodedata.normalize("NFKC", raw_text)))
        return min(keyword_count + money_patterns, 5)

    @staticmethod
    def _host_matches(domain: str, expected: str) -> bool:
        return domain == expected or domain.endswith(f".{expected}")

    def get_all_features(self, text: str) -> Dict[str, int]:
        """Extract features while keeping raw amounts and Unicode hosts intact."""
        raw_text = re.sub(r"\[(?:TYPE|TITLE|CONTENT)\](?:\s*(?:SMS|E-mail))?", " ", str(text))
        unwrapped_text = remove_generator_url_wrappers(raw_text)
        keyword_text = self._normalize_for_keywords(unwrapped_text)
        urls = self._extract_urls(raw_text)

        features = {
            "urgency_score": self._count_triggers(keyword_text, URGENCY_KEYWORD),
            "threat_score": self._count_triggers(keyword_text, THREAT_KEYWORD),
            "verif_score": self._count_triggers(keyword_text, VERIFICATION_KEYWORD),
            "action_score": self._count_triggers(keyword_text, ACTION_KEYWORDS),
            "fin_score": self._get_financial_index(keyword_text, unwrapped_text, FINANCIAL_KEYWORDS),
            "emo_score": self._calculate_emotionality(raw_text),
            "num_urls": len(urls),
            "has_suspicious_tld": 0,
            "has_homograph_attack": 0,
            "has_idn_domain": 0,
            "has_url_shortener": 0,
        }

        for url in urls:
            domain = self._get_domain(url)
            if not domain:
                continue

            suffix = domain.rsplit(".", maxsplit=1)[-1]

            if suffix in SUSPICIOUS_TLDS:
                features["has_suspicious_tld"] = 1

            if any(self._host_matches(domain, shortener) for shortener in URL_SHORTENERS):
                features["has_url_shortener"] = 1

            if "xn--" in domain or not domain.isascii():
                features["has_idn_domain"] = 1

            if self._check_homograph(domain):
                features["has_homograph_attack"] = 1

        return features
