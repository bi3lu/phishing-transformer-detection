import random
import re

from src.data.adversarial_augment.config import CALL_TO_ACTIONS, FINANCIAL_TRAPS, SAFE_ANCHORS


class AdversarialAugmenter:

    _regex = re.compile(r"https?://[^\s]+")
    _money_regex = re.compile(r"(\d+[\.,]\d{2}\s?(?:PLN|zł|eur|usd|€|\$))", re.IGNORECASE)

    @classmethod
    def generate_hard_phish(cls, text: str) -> str:
        text_no_link = cls._regex.sub("", text)

        if not cls._money_regex.search(text_no_link):
            fake_amount = f"{random.randint(1, 50)}.{random.randint(10, 99)} PLN"
            text_no_link += f" {random.choice(FINANCIAL_TRAPS)} {fake_amount}"

        anchor = random.choice(SAFE_ANCHORS)
        cta = random.choice(CALL_TO_ACTIONS)

        return f"{text_no_link} {cta}\n\n{anchor}"
