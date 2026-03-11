import random
import re

from src.data.adversarial_augment.config import CALL_TO_ACTIONS, SAFE_ANCHORS


class AdversarialAugmenter:
    """
    Provides methods for generating adversarial phishing examples.

    This class focuses on creating 'hard' phishing samples by removing
    technical indicators like URLs and injecting deceptive elements
    such as safe footers and behavioral call-to-actions.
    """

    _regex = re.compile(r"https?://[^\s]+")

    @classmethod
    def generate_hard_phish(cls, text: str) -> str:
        """
        Transform a standard phishing text into a linkless adversarial example.

        This method removes all HTTP/HTTPS URLs from the text and appends
        a legitimate-looking anchor (footer). If a link was removed, it also
        injects a behavioral call-to-action to simulate social engineering
        attacks that don't rely on technical infrastructure.

        Args:
            text (str): The original phishing email content.

        Returns:
            str: The modified adversarial text with links removed and
                deceptive context added.
        """
        text_no_link = cls._regex.sub("", text)
        anchor = random.choice(SAFE_ANCHORS)

        if len(text_no_link) < len(text):
            cta = random.choice(CALL_TO_ACTIONS)
            return f"{text_no_link} {cta}\n\n{anchor}"

        return f"{text_no_link}\n\n{anchor}"
