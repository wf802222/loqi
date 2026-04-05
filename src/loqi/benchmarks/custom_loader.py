"""Loader for the custom Loqi trigger micro-benchmark.

Reads YAML scenario files and markdown memory files from data/custom_benchmark/.
Returns typed dataclasses for the evaluation harness.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TriggerExpectation:
    """A single expected trigger (should fire or should not fire)."""

    memory: str
    section: str = ""
    reason: str = ""


@dataclass(frozen=True)
class TriggerScenario:
    """A single trigger recall/precision test case."""

    id: str
    name: str
    category: str
    context: str
    expected_triggers: list[TriggerExpectation] = field(default_factory=list)
    expected_non_triggers: list[TriggerExpectation] = field(default_factory=list)
    baseline_flat_rag: str = ""


@dataclass(frozen=True)
class PromotionStep:
    """A single step in a trigger promotion sequence."""

    context: str
    expectations: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PromotionScenario:
    """A sequence-based test for Hebbian learning / trigger promotion."""

    id: str
    name: str
    category: str
    description: str
    sequence: list[PromotionStep] = field(default_factory=list)


def load_trigger_scenarios(scenarios_dir: Path) -> list[TriggerScenario]:
    """Load trigger recall and precision scenarios from YAML files."""
    results = []

    for yaml_file in sorted(scenarios_dir.glob("trigger_recall.yaml")) + sorted(
        scenarios_dir.glob("trigger_precision.yaml")
    ):
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for s in data.get("scenarios", []):
            expected = [
                TriggerExpectation(
                    memory=t["memory"],
                    section=t.get("section", ""),
                    reason=t.get("reason", ""),
                )
                for t in s.get("expected_triggers", [])
            ]
            non_expected = [
                TriggerExpectation(
                    memory=t["memory"],
                    section=t.get("section", ""),
                    reason=t.get("reason", ""),
                )
                for t in s.get("expected_non_triggers", [])
            ]
            results.append(TriggerScenario(
                id=s["id"],
                name=s["name"],
                category=s["category"],
                context=s["context"],
                expected_triggers=expected,
                expected_non_triggers=non_expected,
                baseline_flat_rag=s.get("baseline_flat_rag", ""),
            ))

    return results


def load_promotion_scenarios(scenarios_dir: Path) -> list[PromotionScenario]:
    """Load trigger promotion / decay sequence scenarios from YAML."""
    results = []

    for yaml_file in sorted(scenarios_dir.glob("trigger_promotion.yaml")):
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for s in data.get("scenarios", []):
            steps = []
            for step in s.get("sequence", []):
                ctx = step["context"]
                exps = {k: v for k, v in step.items() if k != "context"}
                steps.append(PromotionStep(context=ctx, expectations=exps))

            results.append(PromotionScenario(
                id=s["id"],
                name=s["name"],
                category=s["category"],
                description=s.get("description", ""),
                sequence=steps,
            ))

    return results


def load_memories(memories_dir: Path) -> dict[str, str]:
    """Load all markdown memory files. Returns {filename: content}."""
    memories = {}
    for md_file in sorted(memories_dir.glob("*.md")):
        memories[md_file.name] = md_file.read_text(encoding="utf-8")
    return memories
