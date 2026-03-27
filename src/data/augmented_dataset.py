"""On-the-fly augmenting dataset for transformer training.

Wraps training data so that each epoch sees different augmentation
variants, improving model generalisation compared to static,
one-time augmentation stored in CSV.
"""

import random
from typing import Any, Dict

import torch
from torch.utils.data import Dataset

from src.data.adversarial_augment.adversarial_generator import AdversarialAugmenter
from src.data.augment.augment_data import PhishingAugmenter
from src.features.extractor import PhishingFeatureExtractor

FEATURE_DROPOUT_PROB = 0.3


class AugmentedPhishingDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch Dataset that applies text augmentation on-the-fly.

    On each ``__getitem__`` call the content portion of the text is
    re-augmented and feature tags are recalculated, so every epoch
    presents different variants of the training samples.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int = 512,
        augment: bool = True,
        aug_prob: float = 0.1,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

        self.augmenter = PhishingAugmenter(aug_prob=aug_prob)
        self.extractor = PhishingFeatureExtractor()

    def __len__(self) -> int:
        return len(self.texts)

    def _build_augmented_text(self, text: str, is_phishing: bool) -> str:
        """Parse text, augment content, recalculate features, rebuild."""
        # Split into meta (TYPE/SENDER) and content:
        parts = text.split("[CONTENT] ", 1)
        # Strip any existing [FEAT: ...] tag from the meta portion:
        meta = parts[0]
        if "[FEAT:" in meta:
            # Remove the FEAT line – it will be recalculated:
            feat_end = meta.find("]") + 1
            meta = meta[feat_end:].lstrip("\n")

        content = parts[1] if len(parts) > 1 else ""

        # Apply augmentation to phishing content:
        if self.augment and is_phishing:
            if random.random() < 0.2:
                content = AdversarialAugmenter.generate_hard_phish(content)
            else:
                content = self.augmenter.augment(content)

        # Recalculate features from (possibly augmented) content:
        features = self.extractor.get_all_features(content)

        feat_tags = (
            f"[FEAT: URG={features['urgency_score']} "
            f"THR={features['threat_score']} "
            f"VER={features['verif_score']} "
            f"ACT={features['action_score']} "
            f"FIN={features['fin_score']} "
            f"EMO={features['emo_score']} "
            f"TLD={features['has_suspicious_tld']} "
            f"HOMO={features['has_homograph_attack']}]"
        )

        # Feature dropout (training only):
        if self.augment and random.random() < FEATURE_DROPOUT_PROB:
            return f"{meta}[CONTENT] {content}"

        return f"{feat_tags}\n{meta}[CONTENT] {content}"

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        if self.augment:
            text = self._build_augmented_text(text, is_phishing=bool(label))

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(label, dtype=torch.long),
        }
