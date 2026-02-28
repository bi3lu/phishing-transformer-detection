import random
import re
from typing import List


class PhishingAugmenter:
    def __init__(self, aug_prob: float = 0.3):
        self.aug_prob = aug_prob
        self.homoglyphs = {
            "a": ["а", "@"],
            "o": ["0", "о"],
            "e": ["е"],
            "i": ["1", "l", "!"],
            "s": ["5", "$"],
            "p": ["р"],
        }

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
            if char.lower() in self.homoglyphs and random.random() < (self.aug_prob * 0.5):
                new_text += random.choice(self.homoglyphs[char.lower()])

            else:
                new_text += char

        return new_text

    def mask_shortcuts(self, text: str) -> str:
        shortcuts = [
            "InPost",
            "DPD",
            "mBank",
            "PKO",
            "Allegro",
            "Netflix",
        ]  # TODO: Expand and extracto to separate file

        for brand in shortcuts:
            if random.random() < self.aug_prob:
                text = re.sub(brand, "[BRAND]", text, flags=re.IGNORECASE)

        return text

    def augment(self, text: str) -> str:
        text = self.mask_shortcuts(text)
        text = self.introduce_typos(text)
        text = self.apply_homoglyphs(text)

        return text
