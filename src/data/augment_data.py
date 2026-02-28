import random
import re

from src.data.augment_config import HOMOGLYPHS, SHORTCUTS


class PhishingAugmenter:
    def __init__(self, aug_prob: float = 0.3):
        self.aug_prob = aug_prob

    def introduce_typos(self, text: str) -> str:
        words = text.split()

        for i in range(len(words)):
            if len(words[i]) > 4 and random.random() < self.aug_prob:
                idx = random.randint(0, len(words[i]) - 2)
                w_list = list(words[i])
                w_list[idx], w_list[idx + 1] = w_list[idx + 1], w_list[idx]

                words[i] = "".join(w_list)
        return " ".join(words)

    def apply_homoglyphs(self, text: str) -> str:
        new_text = ""

        for char in text:
            if char.lower() in HOMOGLYPHS and random.random() < (self.aug_prob * 0.5):
                new_text += random.choice(HOMOGLYPHS[char.lower()])

            else:
                new_text += char

        return new_text

    def mask_shortcuts(self, text: str) -> str:
        for brand in SHORTCUTS:
            if random.random() < self.aug_prob:
                text = re.sub(brand, "[BRAND]", text, flags=re.IGNORECASE)

        return text

    def augment(self, text: str) -> str:
        text = self.mask_shortcuts(text)
        text = self.introduce_typos(text)
        text = self.apply_homoglyphs(text)

        return text
