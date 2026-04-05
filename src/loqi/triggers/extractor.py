"""Trigger extraction from memory documents.

At write time, the system reads a memory file and extracts trigger patterns.
Each section of a markdown file becomes a separate trigger with:
  - Keywords extracted from the content (important nouns/phrases)
  - A semantic embedding of the full section text
  - The associated document ID for injection when the trigger fires

This is the "write-time encoding" from Loqi's architecture — the thing
that makes triggers work. Standard RAG systems skip this step entirely
and only look for matches at query time.
"""

from __future__ import annotations

import re

import numpy as np

from loqi.graph.embeddings import EmbeddingModel
from loqi.graph.models import Trigger, TriggerOrigin


# Common words that shouldn't be trigger keywords
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "also",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "its", "this", "that", "these", "those",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "if", "because", "while", "although", "though", "unless", "until",
    "about", "up", "out", "off", "over", "down",
})


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text.

    Filters out common stop words and short tokens.
    Keeps domain-specific terms that are likely to be useful trigger patterns.
    """
    # Tokenize: split on whitespace and punctuation
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())

    # Filter
    keywords = []
    seen = set()
    for token in tokens:
        if token in _STOP_WORDS:
            continue
        if len(token) < 3:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)

    return keywords


def _split_markdown_sections(content: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs by ## headings.

    Returns the document-level content (before first ##) as a section too,
    using the # heading as its title if present.
    """
    lines = content.split("\n")
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("## "):
            # Save previous section if it has content
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_heading, body))
            current_heading = line[3:].strip()
            current_lines = []
        elif line.startswith("# ") and not current_heading and not current_lines:
            # Top-level heading — use as title for pre-section content
            current_heading = line[2:].strip()
        else:
            current_lines.append(line)

    # Don't forget the last section
    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, body))

    return sections


def _extract_trigger_keywords(heading: str, body: str) -> list[str]:
    """Extract focused trigger keywords from a section.

    Strategy: heading words are high-signal (they name the topic),
    so they're always included. Body words are filtered more
    aggressively to keep only domain terms, not narrative filler.

    We also extract capitalized terms and technical patterns
    (acronyms, snake_case, etc.) as they tend to be domain-specific.
    """
    keywords = []
    seen = set()

    def _add(word: str):
        w = word.lower()
        if w not in seen and w not in _STOP_WORDS and len(w) >= 3:
            seen.add(w)
            keywords.append(w)

    # Heading words: ALL included (high signal)
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", heading):
        _add(token)

    # Body: extract technical terms, acronyms, and key phrases
    body_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", body)
    for token in body_tokens:
        # Always keep: acronyms (ALL CAPS, 2+ chars)
        if token.isupper() and len(token) >= 2:
            _add(token)
        # Always keep: technical terms (snake_case, hyphenated)
        elif "_" in token or "-" in token:
            _add(token)
        # Always keep: capitalized terms (proper nouns, tech names)
        elif token[0].isupper() and not token.isupper():
            _add(token)

    # Also pull in the first sentence's non-stop words as "what this rule is about"
    first_sentence = body.split(".")[0] if "." in body else body
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", first_sentence.lower()):
        _add(token)

    return keywords


def _is_conversational(content: str) -> bool:
    """Detect if content is conversational (chat turns) vs structured markdown."""
    lines = content.split("\n")[:20]
    chat_markers = sum(1 for l in lines if l.startswith("[user]:") or l.startswith("[assistant]:"))
    return chat_markers >= 2


def _extract_conversational_triggers(
    doc_id: str,
    content: str,
    embedding_model: EmbeddingModel,
) -> list[Trigger]:
    """Extract triggers from conversational text (chat sessions).

    Strategy: extract keywords from USER turns only (the user's
    preferences and requests, not the assistant's responses).
    Focus on named entities, specific preferences, and technical
    terms. Keep the pattern tight — conversational text is noisy.
    """
    lines = content.split("\n")

    # Collect user turn text
    user_text_parts = []
    for line in lines:
        if line.startswith("[user]:"):
            user_text_parts.append(line[7:].strip())

    if not user_text_parts:
        return []

    user_text = " ".join(user_text_parts)

    # Extract tight keywords from user turns only
    keywords = []
    seen = set()

    def _add(word: str):
        w = word.lower()
        if w not in seen and w not in _STOP_WORDS and len(w) >= 3:
            seen.add(w)
            keywords.append(w)

    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", user_text)
    for token in tokens:
        # Capitalized terms (product names, proper nouns)
        if token[0].isupper() and not token.isupper():
            _add(token)
        # Acronyms
        elif token.isupper() and len(token) >= 2:
            _add(token)
        # Technical terms
        elif "_" in token or "-" in token:
            _add(token)

    # Also add non-stop words from the first user turn (highest signal)
    first_turn = user_text_parts[0] if user_text_parts else ""
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", first_turn.lower()):
        _add(token)

    if not keywords:
        return []

    # Compute embedding from user turns only (not assistant responses)
    embedding = embedding_model.encode_single(user_text[:1000])

    return [Trigger(
        id=f"trigger_{doc_id}_conv",
        pattern=keywords,
        pattern_embedding=embedding,
        associated_node_id=doc_id,
        confidence=1.0,
        origin=TriggerOrigin.EXPLICIT,
    )]


def extract_triggers(
    doc_id: str,
    content: str,
    embedding_model: EmbeddingModel,
) -> list[Trigger]:
    """Extract triggers from a document (markdown or conversational).

    Automatically detects the format and uses the appropriate
    extraction strategy:
    - Markdown with ## headers: extract per-section triggers
    - Conversational text ([user]/[assistant] turns): extract from
      user turns only with tight keyword focus

    Args:
        doc_id: The document ID this trigger is associated with.
        content: The text content to extract triggers from.
        embedding_model: For computing semantic embeddings.

    Returns:
        List of Trigger objects ready to store in the GraphStore.
    """
    # Detect format and dispatch
    if _is_conversational(content):
        return _extract_conversational_triggers(doc_id, content, embedding_model)

    # Structured markdown path
    sections = _split_markdown_sections(content)
    triggers = []

    for i, (heading, body) in enumerate(sections):
        keywords = _extract_trigger_keywords(heading, body)

        if not keywords:
            continue

        # Compute semantic embedding of the full section
        full_text = f"{heading} {body}" if heading else body
        embedding = embedding_model.encode_single(full_text)

        trigger_id = f"trigger_{doc_id}_{i}"

        triggers.append(Trigger(
            id=trigger_id,
            pattern=keywords,
            pattern_embedding=embedding,
            associated_node_id=doc_id,
            confidence=1.0,
            origin=TriggerOrigin.EXPLICIT,
        ))

    return triggers
