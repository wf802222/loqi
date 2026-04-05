"""Pipeline configuration for ablation studies.

PipelineConfig controls which layers are active and their hyperparameters.
Each ablation variant is just a different config — the pipeline code checks
config flags to decide what to run.

Standard ablation variants (ADR-004):
  - flat-rag:          graph=False, triggers=False, diffuse=False, hebbian=False
  - graph-only:        graph=True,  triggers=False, diffuse=False, hebbian=False
  - loqi-no-triggers:  graph=True,  triggers=False, diffuse=True,  hebbian=True
  - loqi-no-diffuse:   graph=True,  triggers=True,  diffuse=False, hebbian=True
  - loqi-no-hebbian:   graph=True,  triggers=True,  diffuse=True,  hebbian=False
  - loqi-full:         graph=True,  triggers=True,  diffuse=True,  hebbian=True
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a Loqi pipeline variant."""

    # --- Layer toggles (ablation) ---
    enable_graph: bool = True
    enable_triggers: bool = True
    enable_diffuse: bool = True
    enable_hebbian: bool = True
    enable_llm_gate: bool = False  # v2.5: local LLM trigger suppression

    # --- Retrieval parameters ---
    focused_top_k: int = 10
    focused_max_depth: int = 3
    diffuse_top_k: int = 5
    diffuse_temperature: float = 1.0
    diffuse_novelty_penalty: float = 0.5

    # --- Trigger parameters ---
    trigger_confidence_threshold: float = 0.15
    trigger_max_injections: int = 5

    # --- Hebbian parameters ---
    hebbian_strengthen_rate: float = 0.1
    hebbian_decay_rate: float = 0.02
    hebbian_promotion_threshold_soft: int = 3
    hebbian_promotion_threshold_hard: int = 6
    hebbian_promotion_threshold_trigger: int = 10

    # --- Reproducibility ---
    random_seed: int = 42

    @property
    def variant_name(self) -> str:
        """Generate a human-readable name for this configuration."""
        if not self.enable_graph:
            return "flat-rag"

        parts = []
        if not self.enable_triggers:
            parts.append("no-triggers")
        if not self.enable_diffuse:
            parts.append("no-diffuse")
        if not self.enable_hebbian:
            parts.append("no-hebbian")

        if not parts:
            return "loqi-full"
        if len(parts) == 3:
            return "graph-only"
        return "loqi-" + "-".join(parts)


# --- Pre-built ablation variants ---

FLAT_RAG = PipelineConfig(
    enable_graph=False,
    enable_triggers=False,
    enable_diffuse=False,
    enable_hebbian=False,
)

GRAPH_ONLY = PipelineConfig(
    enable_graph=True,
    enable_triggers=False,
    enable_diffuse=False,
    enable_hebbian=False,
)

LOQI_NO_TRIGGERS = PipelineConfig(
    enable_graph=True,
    enable_triggers=False,
    enable_diffuse=True,
    enable_hebbian=True,
)

LOQI_NO_DIFFUSE = PipelineConfig(
    enable_graph=True,
    enable_triggers=True,
    enable_diffuse=False,
    enable_hebbian=True,
)

LOQI_NO_HEBBIAN = PipelineConfig(
    enable_graph=True,
    enable_triggers=True,
    enable_diffuse=True,
    enable_hebbian=False,
)

LOQI_FULL = PipelineConfig(
    enable_graph=True,
    enable_triggers=True,
    enable_diffuse=True,
    enable_hebbian=True,
)

ALL_VARIANTS = {
    "flat-rag": FLAT_RAG,
    "graph-only": GRAPH_ONLY,
    "loqi-no-triggers": LOQI_NO_TRIGGERS,
    "loqi-no-diffuse": LOQI_NO_DIFFUSE,
    "loqi-no-hebbian": LOQI_NO_HEBBIAN,
    "loqi-full": LOQI_FULL,
}
