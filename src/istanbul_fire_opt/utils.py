"""Shared utility functions."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any


TURKISH_TRANSLATION = str.maketrans(
    {
        "ç": "c",
        "Ç": "c",
        "ğ": "g",
        "Ğ": "g",
        "ı": "i",
        "I": "i",
        "İ": "i",
        "ö": "o",
        "Ö": "o",
        "ş": "s",
        "Ş": "s",
        "ü": "u",
        "Ü": "u",
    }
)


def normalize_key(value: Any) -> str:
    """Return a stable ASCII key for Turkish district names."""

    if value is None:
        return ""
    text = str(value).strip().translate(TURKISH_TRANSLATION).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_case_turkish(value: Any) -> str:
    """Readable district name for tables and charts."""

    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return " ".join(word[:1].upper() + word[1:].lower() for word in text.split())


def as_float(value: Any, default: float | None = math.nan) -> float:
    """Parse numeric values from raw XLSX/CSV cells."""

    if value is None or value == "":
        return default if default is not None else math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return default if default is not None else math.nan


def weighted_percentile(values: Any, weights: Any, percentile: float) -> float:
    """Weighted percentile for one-dimensional arrays."""

    import numpy as np

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cum_weights = np.cumsum(sorted_weights)
    threshold = percentile / 100.0 * cum_weights[-1]
    return float(sorted_values[np.searchsorted(cum_weights, threshold, side="left")])

