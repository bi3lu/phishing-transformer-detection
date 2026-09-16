"""Label-blind template clustering with inspectable similarity edges."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CONTENT_COL, TEMPLATE_GROUP_COL
from src.data.hygiene import normalize_template, template_group_id


def cluster_templates(
    df: pd.DataFrame, overrides_path: Path | None = None, merge_similarity: float = 0.92
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge near-identical wording conservatively; never inspect labels.

    Edges below 0.92 (down to 0.85) are review candidates only. Explicit
    separations block transitive merges, not just direct edges.
    """
    result = df.copy()
    normalized = result[CONTENT_COL].astype(str).map(normalize_template)
    texts = sorted(normalized.unique())
    ids = [template_group_id(text) for text in texts]
    parent = list(range(len(texts)))
    members = {i: {ids[i]} for i in parent}
    overrides: dict[str, Any] = {"merge": [], "separate": []}

    if overrides_path and overrides_path.exists():
        overrides = json.loads(overrides_path.read_text())

    forbidden = {frozenset(pair) for pair in overrides.get("separate", [])}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]

        return i

    def union(i: int, j: int) -> bool:
        a, b = root(i), root(j)

        if a == b:
            return True

        if any(pair <= members[a] | members[b] for pair in forbidden):
            return False

        parent[b] = a
        members[a] |= members.pop(b)
        return True

    for left, right in overrides.get("merge", []):
        if left not in ids or right not in ids or not union(ids.index(left), ids.index(right)):
            raise ValueError(f"Invalid or contradictory template override: {left}, {right}")

    edges = []
    if len(texts) > 1:
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), dtype=np.float32).fit_transform(texts)

        # Chunked sparse products bound memory; no N x N dense allocation:
        for start in range(0, len(texts), 128):
            similarity = (matrix[start : start + 128] @ matrix.T).tocoo()

            for row, col, score in zip(similarity.row, similarity.col, similarity.data):
                left, right = start + int(row), int(col)

                if left >= right or score < 0.85:
                    continue
                merged = score >= merge_similarity and union(left, right)
                edges.append(
                    {
                        "left": ids[left],
                        "right": ids[right],
                        "similarity": float(score),
                        "action": "merged" if merged else "review",
                        "left_text": texts[left],
                        "right_text": texts[right],
                    }
                )

    groups = {text: min(members[root(i)]) for i, text in enumerate(texts)}
    result["Normalized_Template"] = normalized
    result[TEMPLATE_GROUP_COL] = normalized.map(groups)
    return result, pd.DataFrame(edges, columns=["left", "right", "similarity", "action", "left_text", "right_text"])
