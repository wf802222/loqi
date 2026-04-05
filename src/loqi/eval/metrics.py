"""Evaluation metrics for Loqi benchmarks.

Implements the standard metrics used by MuSiQue, HotpotQA, and LongMemEval:
  - Answer F1 and Exact Match (token-level, normalized)
  - Support F1 (set-level, over document IDs)
  - Retrieval Recall@k and Precision@k
  - Trigger recall and precision (for custom benchmark)

All functions are pure — no side effects, no state.
"""

import re
import string
from collections import Counter


# ---------------------------------------------------------------------------
# Text normalization (matches MuSiQue/HotpotQA evaluation scripts)
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison.

    Matches the standard normalization used by SQuAD, HotpotQA, and MuSiQue:
    lowercase, remove articles, remove punctuation, collapse whitespace.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def _get_tokens(text: str) -> list[str]:
    """Tokenize normalized text into words."""
    return normalize_answer(text).split()


# ---------------------------------------------------------------------------
# Answer metrics
# ---------------------------------------------------------------------------

def answer_exact_match(prediction: str, gold: str, aliases: list[str] | None = None) -> float:
    """Exact match after normalization. Returns 1.0 or 0.0.

    Checks against gold answer and all aliases.
    """
    pred_norm = normalize_answer(prediction)
    candidates = [gold] + (aliases or [])
    return 1.0 if any(normalize_answer(c) == pred_norm for c in candidates) else 0.0


def answer_f1(prediction: str, gold: str, aliases: list[str] | None = None) -> float:
    """Token-level F1 between prediction and gold answer.

    Returns the maximum F1 across gold and all aliases.
    """
    pred_tokens = _get_tokens(prediction)
    if not pred_tokens:
        return 0.0

    candidates = [gold] + (aliases or [])
    best_f1 = 0.0

    for candidate in candidates:
        gold_tokens = _get_tokens(candidate)
        if not gold_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


# ---------------------------------------------------------------------------
# Support / retrieval metrics
# ---------------------------------------------------------------------------

def support_f1(predicted_ids: set[str], gold_ids: set[str]) -> float:
    """Set-level F1 over predicted vs gold supporting document IDs.

    This is the standard support metric from HotpotQA and MuSiQue.
    """
    if not predicted_ids and not gold_ids:
        return 1.0
    if not predicted_ids or not gold_ids:
        return 0.0

    common = predicted_ids & gold_ids
    precision = len(common) / len(predicted_ids)
    recall = len(common) / len(gold_ids)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def retrieval_recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Recall@k: fraction of gold documents found in top-k retrieved."""
    if not gold_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & gold_ids) / len(gold_ids)


def retrieval_precision_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Precision@k: fraction of top-k retrieved that are gold documents."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & gold_ids) / len(top_k)


# ---------------------------------------------------------------------------
# Trigger metrics (custom benchmark)
# ---------------------------------------------------------------------------

def trigger_recall(fired_memories: set[str], expected_memories: set[str]) -> float:
    """What fraction of expected triggers actually fired?"""
    if not expected_memories:
        return 1.0
    return len(fired_memories & expected_memories) / len(expected_memories)


def trigger_precision(fired_memories: set[str], expected_memories: set[str],
                      non_expected_memories: set[str]) -> float:
    """What fraction of fired triggers were expected (not false positives)?

    A trigger is a false positive if it fired AND it's in the non_expected set.
    Triggers that fired but aren't in either set are treated as neutral (not penalized).
    """
    if not fired_memories:
        return 1.0
    false_positives = fired_memories & non_expected_memories
    return 1.0 - (len(false_positives) / len(fired_memories))


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def aggregate_metrics(metric_values: list[float]) -> dict[str, float]:
    """Compute mean, min, max over a list of metric values."""
    if not metric_values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    return {
        "mean": sum(metric_values) / len(metric_values),
        "min": min(metric_values),
        "max": max(metric_values),
        "count": len(metric_values),
    }
