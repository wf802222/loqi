"""SmolLM trigger gate -- cheap first-pass relevance filter.

Before a trigger injects a section into retrieval results, SmolLM
evaluates whether the section is actually relevant to the current
task. This catches false positives that pass deterministic guards
but are contextually irrelevant.

The LLM does not create, modify, or delete memory. It only
recommends suppress/allow.
"""

from __future__ import annotations

import time

from loqi.llm.client import OllamaClient


SMOL_MODEL = "smollm2:1.7b"

GATE_PROMPT = """You are a relevance filter. Given a user's current task and a memory section that was triggered, decide if the memory is relevant to the task.

TASK: {query}

TRIGGERED MEMORY:
Title: {section_title}
From document: {parent_doc}
Content: {section_preview}

Is this memory section relevant to the current task?

Respond with ONLY a JSON object:
{{"decision": "irrelevant", "reason": "one sentence"}}
or
{{"decision": "relevant", "reason": "one sentence"}}
or
{{"decision": "possibly_relevant", "reason": "one sentence"}}"""


class TriggerGate:
    """Uses SmolLM to filter trigger candidates for relevance."""

    def __init__(self, client: OllamaClient | None = None, model: str = SMOL_MODEL):
        self._client = client or OllamaClient()
        self._model = model
        self._available = None  # lazy check

    def _check_available(self) -> bool:
        if self._available is None:
            self._available = self._client.is_available()
        return self._available

    def evaluate(
        self,
        query: str,
        section_title: str,
        section_content: str,
        parent_doc: str,
    ) -> dict:
        """Ask SmolLM whether a triggered section is relevant.

        Returns dict with 'decision' (irrelevant/possibly_relevant/relevant),
        'reason', and 'latency_ms'.

        Falls back to 'relevant' (allow) if Ollama is unavailable.
        """
        if not self._check_available():
            return {
                "decision": "relevant",
                "reason": "LLM unavailable, fallback to allow",
                "latency_ms": 0,
            }

        prompt = GATE_PROMPT.format(
            query=query,
            section_title=section_title,
            parent_doc=parent_doc,
            section_preview=section_content[:200],
        )

        t0 = time.time()
        result = self._client.classify(self._model, prompt)
        latency = (time.time() - t0) * 1000

        decision = result.get("decision", "relevant")
        # Normalize to valid values
        if decision not in ("irrelevant", "possibly_relevant", "relevant"):
            decision = "relevant"

        return {
            "decision": decision,
            "reason": result.get("reason", ""),
            "latency_ms": latency,
        }

    def filter_triggers(
        self,
        query: str,
        candidates: dict[str, float],
        section_lookup: dict,
    ) -> tuple[dict[str, float], list[str], float]:
        """Filter a batch of trigger candidates through SmolLM.

        Args:
            query: The current task/context.
            candidates: {section_id: trigger_score} of candidates that
                passed deterministic guards.
            section_lookup: callable or dict to get (title, content, parent)
                for a section ID.

        Returns:
            (surviving_candidates, suppressed_ids, total_latency_ms)
        """
        if not candidates or not self._check_available():
            return candidates, [], 0.0

        surviving = {}
        suppressed = []
        total_latency = 0.0

        for nid, score in candidates.items():
            info = section_lookup.get(nid)
            if info is None:
                surviving[nid] = score
                continue

            title, content, parent = info

            result = self.evaluate(query, title, content, parent)
            total_latency += result["latency_ms"]

            if result["decision"] == "irrelevant":
                suppressed.append(nid)
            else:
                surviving[nid] = score

        return surviving, suppressed, total_latency
