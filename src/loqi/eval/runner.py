"""Evaluation runner — orchestrates benchmark evaluation.

Feeds BenchmarkExamples through a RetrievalSystem, collects RetrievalResults,
and computes metrics. Produces structured results for analysis and reporting.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from loqi.benchmarks.schema import BenchmarkExample, Document
from loqi.eval.metrics import (
    aggregate_metrics,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
    support_f1,
    trigger_precision,
    trigger_recall,
)
from loqi.eval.protocol import RetrievalResult, RetrievalSystem


@dataclass
class ExampleResult:
    """Result for a single benchmark example."""

    example_id: str
    benchmark: str
    category: str
    retrieval_result: RetrievalResult
    metrics: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """Aggregated result for an entire benchmark run."""

    system_name: str
    benchmark_name: str
    example_results: list[ExampleResult] = field(default_factory=list)
    aggregate: dict[str, dict[str, float]] = field(default_factory=dict)
    total_time_s: float = 0.0

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = [
            f"=== {self.system_name} on {self.benchmark_name} ===",
            f"Examples: {len(self.example_results)}",
            f"Total time: {self.total_time_s:.1f}s",
        ]
        for metric_name, agg in sorted(self.aggregate.items()):
            lines.append(
                f"  {metric_name}: {agg['mean']:.4f}"
                f" (min={agg['min']:.4f}, max={agg['max']:.4f})"
            )
        return "\n".join(lines)


def evaluate_retrieval(
    system: RetrievalSystem,
    examples: list[BenchmarkExample],
    top_k: int = 10,
    reset_between: bool = True,
) -> BenchmarkResult:
    """Run a retrieval system against a list of benchmark examples.

    Args:
        system: The system to evaluate.
        examples: Benchmark examples with ground truth.
        top_k: Number of documents to retrieve per query.
        reset_between: If True, reset system state between examples.
            Set False for Hebbian learning tests where state accumulates.

    Returns:
        BenchmarkResult with per-example and aggregate metrics.
    """
    example_results = []
    metric_lists: dict[str, list[float]] = {}
    start_time = time.time()

    for ex in examples:
        if reset_between:
            system.reset()

        # Index the example's documents
        system.index(ex.documents)

        # Retrieve
        t0 = time.time()
        result = system.retrieve(ex.query, top_k=top_k)
        latency_ms = (time.time() - t0) * 1000

        # Compute metrics
        gold_ids = set(ex.supporting_doc_ids)
        metrics: dict[str, float] = {}

        # Retrieval metrics
        metrics["support_f1"] = support_f1(set(result.retrieved_ids), gold_ids)
        for k in [1, 3, 5, 10]:
            if k <= top_k:
                metrics[f"recall@{k}"] = retrieval_recall_at_k(
                    result.retrieved_ids, gold_ids, k
                )
                metrics[f"precision@{k}"] = retrieval_precision_at_k(
                    result.retrieved_ids, gold_ids, k
                )

        # Trigger metrics (if system produces triggers and example has expectations)
        if result.triggered_memories:
            expected = set(ex.supporting_doc_ids)
            metrics["trigger_recall"] = trigger_recall(
                result.triggered_memories, expected
            )

        # Hebbian feedback: tell the system which documents were useful
        # Uses ground truth supporting docs as the "usefulness" signal.
        # In production, this would come from LLM synthesis marking
        # which documents actually contributed to the answer.
        if not reset_between and gold_ids:
            system.update(ex.query, result, gold_ids)

        # Accumulate for aggregation
        for name, value in metrics.items():
            metric_lists.setdefault(name, []).append(value)

        example_results.append(ExampleResult(
            example_id=ex.id,
            benchmark=ex.benchmark,
            category=ex.category,
            retrieval_result=result,
            metrics=metrics,
            latency_ms=latency_ms,
        ))

    total_time = time.time() - start_time

    # Aggregate
    aggregate = {
        name: aggregate_metrics(values) for name, values in metric_lists.items()
    }

    benchmark_name = examples[0].benchmark if examples else "unknown"

    return BenchmarkResult(
        system_name=system.name,
        benchmark_name=benchmark_name,
        example_results=example_results,
        aggregate=aggregate,
        total_time_s=total_time,
    )


def evaluate_trigger_scenarios(
    system: RetrievalSystem,
    scenarios: list,
    memories: dict[str, str],
) -> dict[str, dict]:
    """Evaluate a system against the custom trigger benchmark.

    Args:
        system: Must implement triggers (triggered_memories in RetrievalResult).
        scenarios: TriggerScenario objects from custom_loader.
        memories: {filename: content} from custom_loader.load_memories().

    Returns:
        Dict with per-scenario results and aggregate trigger metrics.
    """
    # Index all memories as documents
    docs = [
        Document(id=name, title=name, text=content)
        for name, content in memories.items()
    ]
    system.reset()
    system.index(docs)

    results = []
    recall_values = []
    precision_values = []

    for scenario in scenarios:
        result = system.retrieve(scenario.context, top_k=len(docs))

        expected = {t.memory for t in scenario.expected_triggers}
        non_expected = {t.memory for t in scenario.expected_non_triggers}

        r = trigger_recall(result.triggered_memories, expected)
        p = trigger_precision(result.triggered_memories, expected, non_expected)

        recall_values.append(r)
        precision_values.append(p)

        results.append({
            "id": scenario.id,
            "name": scenario.name,
            "category": scenario.category,
            "trigger_recall": r,
            "trigger_precision": p,
            "fired": sorted(result.triggered_memories),
            "expected": sorted(expected),
            "non_expected": sorted(non_expected),
        })

    return {
        "system": system.name,
        "scenarios": results,
        "aggregate": {
            "trigger_recall": aggregate_metrics(recall_values),
            "trigger_precision": aggregate_metrics(precision_values),
        },
    }


def save_results(result: BenchmarkResult, output_dir: Path) -> Path:
    """Save benchmark results to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result.system_name}_{result.benchmark_name}.json"
    path = output_dir / filename

    data = {
        "system_name": result.system_name,
        "benchmark_name": result.benchmark_name,
        "total_time_s": result.total_time_s,
        "aggregate": result.aggregate,
        "examples": [
            {
                "example_id": er.example_id,
                "benchmark": er.benchmark,
                "category": er.category,
                "metrics": er.metrics,
                "latency_ms": er.latency_ms,
                "retrieved_ids": er.retrieval_result.retrieved_ids,
                "triggered_memories": sorted(
                    er.retrieval_result.triggered_memories
                ),
            }
            for er in result.example_results
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path
