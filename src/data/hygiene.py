"""Deterministic data-cleaning and leakage-prevention utilities.

The synthetic generators occasionally return search-engine wrappers around
URLs and repeated templates with only identifiers or amounts changed.  This
module removes the transport artefacts and assigns a stable template group so
related messages can never be split across train, validation, and test.
"""

import hashlib
import re
import unicodedata
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

from src.config import CONTENT_COL, LABEL_COL, TEMPLATE_GROUP_COL

GOOGLE_SEARCH_URL_RE = re.compile(r"https?://(?:www\.)?google\.com/search\?[^\s)]+", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s)\]]+|\b[a-z0-9][a-z0-9.-]*\.(?:pl|com|net|org|eu|info|xyz|online|tk|ga|cf|gq)(?:/[^\s)\]]*)?",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
IDENTIFIER_RE = re.compile(r"\b(?=[\w-]*\d)(?=[\w-]*[^\W\d_])[\w-]+\b", re.UNICODE)
MONTH_RE = re.compile(
    r"\b(?:stycz(?:nia|en|eń)|lut(?:ego|y)|mar(?:ca|zec)|kwie(?:tnia|cien|cień)|ma[jy]a?|"
    r"czerw(?:ca|iec)|lip(?:ca|iec)|sierp(?:nia|ien|ień)|wrze(?:snia|śnia|sien|sień)|"
    r"pa[źz]dziernik(?:a)?|listopad(?:a)?|grud(?:nia|zien|zień))\b"
)
BARE_URL_RE = re.compile(
    r"(?<![@\w.-])(?:[\w-]+\.)+(?:[^\W\d_]{2,63}|xn--[a-z0-9-]+)(?:/[^\s)\]]*)?",
    re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"\s+")


def _google_query_target(url: str) -> str:
    """Return the intended URL/text from a Google search wrapper."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query).get("q", [""])[0]
    return unquote(query).strip() or url


def remove_generator_url_wrappers(text: str) -> str:
    """Replace synthetic Google/Markdown wrappers with their displayed URL.

    A record such as ``[bank.example](google.com/search?q=https://bank.example)``
    becomes ``https://bank.example``.  The query target is retained so the
    actual host remains available to security analysis, while the synthetic
    search-engine wrapper is removed.
    """

    def replace_markdown(match: re.Match[str]) -> str:
        display, target = match.group(1).strip(), match.group(2)

        if GOOGLE_SEARCH_URL_RE.fullmatch(target):
            return _google_query_target(target) or display

        return match.group(0)

    cleaned = MARKDOWN_LINK_RE.sub(replace_markdown, str(text))
    cleaned = GOOGLE_SEARCH_URL_RE.sub(lambda match: _google_query_target(match.group(0)), cleaned)
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def normalize_template(text: str) -> str:
    """Normalize variable entities while retaining the linguistic template."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = remove_generator_url_wrappers(normalized)
    normalized = EMAIL_RE.sub(" <EMAIL> ", normalized)
    normalized = URL_RE.sub(" <URL> ", normalized)
    normalized = BARE_URL_RE.sub(" <URL> ", normalized)
    normalized = MONTH_RE.sub(" <MONTH> ", normalized)
    normalized = IDENTIFIER_RE.sub(" <ID> ", normalized)
    normalized = NUMBER_RE.sub(" <NUM> ", normalized)
    normalized = re.sub(r"[^\w<>]+", " ", normalized, flags=re.UNICODE)
    return WHITESPACE_RE.sub(" ", normalized).strip()


def template_group_id(text: str) -> str:
    """Create a stable, non-reversible ID for a normalized template."""
    return hashlib.sha256(normalize_template(text).encode("utf-8")).hexdigest()[:20]


def add_template_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with deterministic template-group identifiers."""
    if CONTENT_COL not in df.columns:
        raise ValueError(f"Missing required column: {CONTENT_COL}")

    result = df.copy()
    result[TEMPLATE_GROUP_COL] = result[CONTENT_COL].astype(str).map(template_group_id)
    return result


def assert_group_label_consistency(df: pd.DataFrame) -> None:
    """Reject contradictory *identical observable messages*, not mixed families.

    Related legitimate and malicious variants may correctly have different
    labels. They must stay in one split without changing their ground truth.
    """
    keys = [c for c in ("Type", "Title", CONTENT_COL) if c in df.columns]
    conflicts = df.groupby(keys, dropna=False)[LABEL_COL].nunique(dropna=False)
    conflicting_groups = conflicts[conflicts > 1].index.tolist()

    if conflicting_groups:
        preview = str(conflicting_groups[:5])
        raise ValueError(
            f"Found {len(conflicting_groups)} identical messages with conflicting labels: {preview}. "
            "Resolve them manually before splitting."
        )


def exact_duplicate_mask(df: pd.DataFrame, subset: Iterable[str]) -> pd.Series:
    """Identify all but the first deterministic occurrence of exact records."""
    columns = [column for column in subset if column in df.columns]

    if not columns:
        raise ValueError("No duplicate-key columns are present in the dataframe")

    return df.duplicated(subset=columns, keep="first")
