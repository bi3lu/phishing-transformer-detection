"""Optional label-agnostic surface-perturbation dataset.

The canonical experiment does not use augmentation.  This class exists only
for an explicitly reported ablation and deliberately applies the same
semantics-preserving operations to both labels.  It never injects feature tags,
brand placeholders, calls to action, amounts, or synthetic footers.
"""

from typing import Any, Dict

import torch
from torch.utils.data import Dataset

from src.config import DEFAULT_MAX_LENGTH
from src.data.augment.augment_data import PhishingAugmenter


class AugmentedPhishingDataset(Dataset[Dict[str, torch.Tensor]]):
    """Apply identical typo/homoglyph perturbations to both classes."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int = DEFAULT_MAX_LENGTH,
        augment: bool = False,
        aug_prob: float = 0.1,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have equal lengths")
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.augmenter = PhishingAugmenter(aug_prob=aug_prob)

    def __len__(self) -> int:
        return len(self.texts)

    def _build_augmented_text(self, text: str) -> str:
        """Perturb only content, preserving TYPE/TITLE and the input schema."""
        parts = text.split("[CONTENT] ", 1)

        if len(parts) != 2:
            return text

        metadata, content = parts
        # Brand masking creates a synthetic [BRAND] token, while adversarial
        # augmentation changes the class semantics.  Neither is used here.
        content = self.augmenter.introduce_typos(content)
        content = self.augmenter.apply_homoglyphs(content)
        return f"{metadata}[CONTENT] {content}"

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        if self.augment:
            text = self._build_augmented_text(text)

        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length)
        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
