"""Learning curve evaluation for Hebbian learning.

Demonstrates that Loqi improves over time with Hebbian learning ON.
Runs a sequence of queries against a persistent graph (no reset between
examples) and plots retrieval quality at each step.

The key comparison: learning-ON curve should slope upward while
learning-OFF stays flat.

Usage: python scripts/run_learning_curve.py [--n-examples 50]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

from loqi.benchmarks.loaders import load_musique
from loqi.eval.runner import evaluate_retrieval
from loqi.graph.embeddings import EmbeddingModel
from loqi.pipeline.config import LOQI_FULL, LOQI_NO_HEBBIAN
from loqi.retrieval.graph_rag import GraphRAG


def running_average(values, window=10):
    """Compute running average with given window size."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result


def main():
    parser = argparse.ArgumentParser(description="Run Hebbian learning curve")
    parser.add_argument("--n-examples", type=int, default=50)
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    model = EmbeddingModel()

    # Load examples -- use train sample for learning, not validation
    all_examples = load_musique(data_dir / "raw" / "musique" / "train_sample.jsonl")
    answerable = [e for e in all_examples if e.metadata.get("answerable", True)]
    examples = answerable[:args.n_examples]
    print(f"Using {len(examples)} answerable examples for learning curve")

    # --- Run with Hebbian learning ON (no reset between examples) ---
    print("\nRunning with Hebbian learning ON...")
    system_on = GraphRAG(config=LOQI_FULL, embedding_model=model)
    result_on = evaluate_retrieval(
        system_on, examples, top_k=5, reset_between=False
    )
    scores_on = [
        er.metrics.get("recall@5", 0) for er in result_on.example_results
    ]

    # --- Run with Hebbian learning OFF (no reset, but no learning) ---
    print("Running with Hebbian learning OFF...")
    system_off = GraphRAG(config=LOQI_NO_HEBBIAN, embedding_model=model)
    result_off = evaluate_retrieval(
        system_off, examples, top_k=5, reset_between=False
    )
    scores_off = [
        er.metrics.get("recall@5", 0) for er in result_off.example_results
    ]

    # --- Run with reset between (stateless baseline) ---
    print("Running stateless baseline (reset between)...")
    system_reset = GraphRAG(config=LOQI_FULL, embedding_model=model)
    result_reset = evaluate_retrieval(
        system_reset, examples, top_k=5, reset_between=True
    )
    scores_reset = [
        er.metrics.get("recall@5", 0) for er in result_reset.example_results
    ]

    # Compute running averages
    avg_on = running_average(scores_on)
    avg_off = running_average(scores_off)
    avg_reset = running_average(scores_reset)

    # Print results
    print(f"\n{'='*60}")
    print(f"LEARNING CURVE RESULTS (n={len(examples)})")
    print(f"{'='*60}")

    print(f"\n{'Query #':<10} {'Learning ON':>12} {'Learning OFF':>13} {'Stateless':>10}")
    print("-" * 47)
    for i in range(0, len(examples), max(1, len(examples) // 10)):
        print(f"{i+1:<10} {avg_on[i]:>12.3f} {avg_off[i]:>13.3f} {avg_reset[i]:>10.3f}")

    # Final averages
    final_on = sum(scores_on) / len(scores_on)
    final_off = sum(scores_off) / len(scores_off)
    final_reset = sum(scores_reset) / len(scores_reset)

    print(f"\n{'Overall':.<20} {final_on:.4f} {'':>6} {final_off:.4f} {'':>3} {final_reset:.4f}")

    # Check if learning curve slopes up
    half = len(scores_on) // 2
    first_half_on = sum(scores_on[:half]) / half
    second_half_on = sum(scores_on[half:]) / (len(scores_on) - half)
    first_half_off = sum(scores_off[:half]) / half
    second_half_off = sum(scores_off[half:]) / (len(scores_off) - half)

    print(f"\nFirst half vs second half:")
    print(f"  Learning ON:  {first_half_on:.4f} -> {second_half_on:.4f} (delta: {second_half_on - first_half_on:+.4f})")
    print(f"  Learning OFF: {first_half_off:.4f} -> {second_half_off:.4f} (delta: {second_half_off - first_half_off:+.4f})")

    # Graph state after learning
    store = system_on._store
    print(f"\nGraph state after {len(examples)} queries (learning ON):")
    print(f"  Nodes: {store.get_node_count()}")
    print(f"  Edges: {store.get_edge_count()}")
    print(f"  Triggers: {store.get_trigger_count()}")
    print(f"  Episodes logged: {len(system_on._episode_log)}")

    # Save results
    output_dir = data_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "learning_curve.json"
    with open(output_path, "w") as f:
        json.dump({
            "n_examples": len(examples),
            "scores_learning_on": scores_on,
            "scores_learning_off": scores_off,
            "scores_stateless": scores_reset,
            "running_avg_on": avg_on,
            "running_avg_off": avg_off,
            "running_avg_reset": avg_reset,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
