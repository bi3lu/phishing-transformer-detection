import re
from difflib import SequenceMatcher
from typing import Dict, List


class PhishingFeatureExtractor:

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
        return re.findall(r'https?://[^\s)"\']+', text.lower())

    def _get_domain(self, url: str) -> str:
        domain = re.sub(r"https?://", "", url)
        return domain.split("/")[0].split(":")[0].split("@")[-1]

    def _check_homograph(self, domain: str) -> bool:
        # 1. ...
        try:
            domain.encode("ascii")
        except UnicodeEncodeError:
            return True

        # 2. ...
        for legit in self.legit_domains:
            legit_name = legit.split(".")[0]
            domain_name = domain.split(".")[0]

            if 0.8 <= SequenceMatcher(None, legit_name, domain_name).ratio() < 1.0:
                return True

        return False

    def _count_triggers(self, text: str, word_list: List[str]) -> int:
        count = 0
        text_lower = text.lower()

        for word in word_list:
            pattern = rf"\b{re.escape(word)}[a-z]*\b"
            count += len(re.findall(pattern, text_lower))

        return min(count, 5)

    def _normalize_for_stats(self, text: str) -> str:
        replacements = {"а": "a", "о": "o", "е": "e", "р": "p", "0": "o", "5": "s", "@": "a"}

        for char, rep in replacements.items():
            text = text.replace(char, rep)

        return text.lower()

    def get_all_features(self, text: str) -> Dict[str, int]:
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
