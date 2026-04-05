"""Ollama HTTP client for local LLM inference.

Wraps Ollama's REST API for simple prompt -> JSON response calls.
Falls back gracefully when Ollama is unavailable.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error


OLLAMA_BASE = "http://localhost:11434"
DEFAULT_TIMEOUT = 30  # seconds


class OllamaClient:
    """Simple client for the local Ollama API."""

    def __init__(self, base_url: str = OLLAMA_BASE, timeout: int = DEFAULT_TIMEOUT):
        self._base = base_url
        self._timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama is running and responding."""
        try:
            req = urllib.request.Request(f"{self._base}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def generate(self, model: str, prompt: str) -> str:
        """Send a prompt to a model and return the raw text response."""
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")

    def classify(self, model: str, prompt: str) -> dict:
        """Send a prompt expecting a JSON response. Parse and return it.

        If the model returns invalid JSON, returns a fallback dict
        with decision="uncertain".
        """
        try:
            raw = self.generate(model, prompt)

            # Try to extract JSON from the response
            # Models sometimes wrap JSON in markdown code blocks
            text = raw.strip()
            if "```" in text:
                # Extract content between code fences
                parts = text.split("```")
                for part in parts[1:]:
                    cleaned = part.strip()
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:].strip()
                    if cleaned.startswith("{"):
                        text = cleaned
                        break

            # Find the JSON object in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])

            return {"decision": "uncertain", "reason": "no JSON in response"}

        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            return {"decision": "uncertain", "reason": str(e)}
