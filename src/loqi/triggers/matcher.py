"""Trigger matching engine.

The always-on scanner from Loqi's architecture. Given incoming context,
determines which stored triggers should fire by combining:
  1. Keyword score — how many trigger keywords appear in the context
  2. Semantic score — cosine similarity between context and trigger embeddings

A trigger fires when its combined score exceeds the confidence threshold.
This runs BEFORE retrieval — it's the pre-retrieval injection step that
makes Loqi different from every other RAG system.
"""

from __future__ import annotations

import re

import numpy as np

from loqi.graph.embeddings import EmbeddingModel, cosine_similarity
from loqi.graph.models import Trigger


def _tokenize_context(context: str) -> set[str]:
    """Tokenize context into lowercase words for keyword matching."""
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", context.lower()))


def keyword_score(trigger: Trigger, context_tokens: set[str]) -> float:
    """Fraction of trigger keywords found in the context.

    Returns a value between 0 and 1. Higher means more keyword overlap.
    """
    if not trigger.pattern:
        return 0.0
    matches = sum(1 for kw in trigger.pattern if kw in context_tokens)
    return matches / len(trigger.pattern)


def semantic_score(
    trigger: Trigger,
    context_embedding: np.ndarray,
) -> float:
    """Cosine similarity between trigger embedding and context embedding.

    Returns a value between -1 and 1. Higher means more semantically similar.
    """
    if trigger.pattern_embedding is None:
        return 0.0
    return cosine_similarity(trigger.pattern_embedding, context_embedding)


def match_triggers(
    triggers: list[Trigger],
    context: str,
    context_embedding: np.ndarray,
    keyword_weight: float = 0.4,
    semantic_weight: float = 0.6,
    threshold: float = 0.3,
) -> list[tuple[Trigger, float]]:
    """Scan all triggers against incoming context.

    Returns triggers that fire, sorted by score descending.
    Each result is a (trigger, combined_score) pair.

    The weighting favors semantics (0.6) over keywords (0.4) because:
    - Keywords catch exact matches but miss paraphrases
    - Semantics catch "fix the modal" as UI work even without the word "UI"
    - But keywords provide precision when semantics are ambiguous

    Args:
        triggers: All stored triggers to scan.
        context: The incoming text context.
        context_embedding: Pre-computed embedding of the context.
        keyword_weight: Weight for keyword score (0-1).
        semantic_weight: Weight for semantic score (0-1).
        threshold: Minimum combined score to fire.

    Returns:
        List of (trigger, score) pairs for triggers that fire.
    """
    context_tokens = _tokenize_context(context)
    fired: list[tuple[Trigger, float]] = []

    for trigger in triggers:
        # Skip triggers below confidence threshold (decayed triggers)
        if trigger.confidence < 0.1:
            continue

        kw = keyword_score(trigger, context_tokens)
        sem = semantic_score(trigger, context_embedding)

        # Combined score, weighted by trigger confidence
        combined = (keyword_weight * kw + semantic_weight * sem) * trigger.confidence

        if combined >= threshold:
            fired.append((trigger, combined))

    # Sort by score descending
    fired.sort(key=lambda x: x[1], reverse=True)
    return fired
