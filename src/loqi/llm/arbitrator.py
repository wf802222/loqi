"""Phi-4 arbitrator -- contextual judgment for ambiguous cases.

Invoked when SmolLM returns 'possibly_relevant' or when retrieval
channels disagree. Phi-4 provides stronger contextual judgment
to decide which section should surface.

The LLM does not create, modify, or delete memory. It only
recommends surface/suppress/rerank.
"""

from __future__ import annotations

import time

from loqi.llm.client import OllamaClient


PHI_MODEL = "phi4:latest"

ARBITRATE_PROMPT = """You are a memory relevance arbitrator. A memory system has retrieved several candidate sections for the user's current task. Some were found by semantic similarity, some by associative triggers, and some by graph connections.

Decide which candidates should be surfaced and which should be suppressed.

CURRENT TASK: {query}

CANDIDATES:
{candidates_text}

For each candidate, decide: surface (show to user) or suppress (hide).
Focus on operational relevance -- is this section genuinely useful for the task at hand, or just topically adjacent?

Respond with ONLY a JSON object:
{{"decisions": [{{"section_id": "...", "action": "surface" or "suppress", "reason": "one sentence"}}], "primary_section_id": "most relevant section ID or null"}}"""


class Arbitrator:
    """Uses Phi-4 for contextual judgment on ambiguous retrieval cases."""

    def __init__(self, client: OllamaClient | None = None, model: str = PHI_MODEL):
        self._client = client or OllamaClient(timeout=60)
        self._model = model
        self._available = None

    def _check_available(self) -> bool:
        if self._available is None:
            self._available = self._client.is_available()
        return self._available

    def arbitrate(
        self,
        query: str,
        candidates: list[dict],
    ) -> dict:
        """Ask Phi-4 to judge a set of retrieval candidates.

        Args:
            query: The current task/context.
            candidates: List of dicts with 'section_id', 'title',
                'content_preview', 'channel' (semantic/trigger/graph).

        Returns:
            Dict with 'decisions' list, 'primary_section_id',
            'latency_ms', and 'suppress_ids'.
        """
        if not self._check_available() or not candidates:
            return {
                "decisions": [],
                "primary_section_id": None,
                "suppress_ids": [],
                "latency_ms": 0,
            }

        # Format candidates for the prompt
        lines = []
        for c in candidates:
            channel = c.get("channel", "unknown")
            lines.append(
                f"- Section: {c['section_id']}\n"
                f"  Title: {c.get('title', 'untitled')}\n"
                f"  Channel: {channel}\n"
                f"  Preview: {c.get('content_preview', '')[:150]}"
            )
        candidates_text = "\n".join(lines)

        prompt = ARBITRATE_PROMPT.format(
            query=query,
            candidates_text=candidates_text,
        )

        t0 = time.time()
        result = self._client.classify(self._model, prompt)
        latency = (time.time() - t0) * 1000

        decisions = result.get("decisions", [])
        suppress_ids = [
            d["section_id"] for d in decisions
            if d.get("action") == "suppress"
        ]

        return {
            "decisions": decisions,
            "primary_section_id": result.get("primary_section_id"),
            "suppress_ids": suppress_ids,
            "latency_ms": latency,
        }
