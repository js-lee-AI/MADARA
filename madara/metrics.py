"""Exact Match and Token F1, matching the paper's evaluation (Section 5.4).

Exact Match is computed after whitespace normalization, lowercasing, and
removal of articles and punctuation. Token F1 is the harmonic mean of
token-level precision and recall. For the binary FEVER task EM and F1 coincide.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation, articles, and redundant whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str | list[str]) -> float:
    """1.0 if the normalized prediction equals any gold answer, else 0.0."""
    golds = [gold] if isinstance(gold, str) else gold
    pred = normalize_answer(prediction)
    return 1.0 if any(pred == normalize_answer(g) for g in golds) else 0.0


def token_f1(prediction: str, gold: str | list[str]) -> float:
    """Best token-level F1 of the prediction against any gold answer."""
    golds = [gold] if isinstance(gold, str) else gold
    return max(_f1_single(prediction, g) for g in golds)


def _f1_single(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
