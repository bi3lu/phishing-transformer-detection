"""Text augmentation strategies for phishing detection datasets.

Provides methods to augment phishing email texts through typos,
homoglyph substitutions, and brand keyword masking.
"""

import random
import re

from src.data.augment_config import HOMOGLYPHS, SHORTCUTS


class PhishingAugmenter:
    """Augments text data through multiple obfuscation techniques.

    Applies realistic transformations to phishing texts to improve model
    robustness, including typos, character substitutions, and keyword masking.
    """

    def __init__(self, aug_prob: float = 0.3):
        """Initialize the augmenter with a given probability.

        Args:
            aug_prob: Probability of applying augmentation to each word.
                Defaults to 0.3.
        """
        self.aug_prob = aug_prob

    def introduce_typos(self, text: str) -> str:
        """Introduce random character transpositions in words.

        For words longer than 4 characters, randomly swaps adjacent characters.

        Args:
            text: Input text to augment.

        Returns:
            Text with random character transpositions applied.
        """
        words = text.split()

        for i in range(len(words)):
            if len(words[i]) > 4 and random.random() < self.aug_prob:
                idx = random.randint(0, len(words[i]) - 2)
                w_list = list(words[i])
                w_list[idx], w_list[idx + 1] = w_list[idx + 1], w_list[idx]

                words[i] = "".join(w_list)
        return " ".join(words)

    def apply_homoglyphs(self, text: str) -> str:
        """Replace characters with visually similar homoglyphs.

        Substitutes characters with lookalike alternatives (e.g., 'a' → 'а'),
        simulating obfuscation techniques used in phishing emails.

        Args:
            text: Input text to augment.

        Returns:
            Text with homoglyph substitutions applied.
        """
        new_text = ""

        for char in text:
            if char.lower() in HOMOGLYPHS and random.random() < (self.aug_prob * 0.5):
                new_text += random.choice(HOMOGLYPHS[char.lower()])

            else:
                new_text += char

        return new_text

    def mask_shortcuts(self, text: str) -> str:
        """Mask brand names with generic placeholder token.

        Replaces known brand shortcuts with [BRAND] token to reduce
        model dependency on specific brand names.

        Args:
            text: Input text to augment.

        Returns:
            Text with brand names masked as [BRAND].
        """
        for brand in SHORTCUTS:
            if random.random() < self.aug_prob:
                text = re.sub(brand, "[BRAND]", text, flags=re.IGNORECASE)

        return text

    def augment(self, text: str) -> str:
        """Apply all augmentation strategies in sequence.

        Applies masking, typo introduction, and homoglyph substitution
        in order to create augmented text samples.

        Args:
            text: Input text to augment.

        Returns:
            Fully augmented text with all transformations applied.
        """
        text = self.mask_shortcuts(text)
        text = self.introduce_typos(text)
        text = self.apply_homoglyphs(text)

        return text
