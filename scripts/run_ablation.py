"""Run the full ablation study across all system variants and benchmarks.

Produces the results table for the white paper. Runs each variant
against MuSiQue (multi-hop retrieval) and the custom trigger benchmark.

Usage: python scripts/run_ablation.py [--n-examples 100]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loqi.benchmarks.custom_loader import load_memories, load_trigger_scenarios
from loqi.benchmarks.loaders import load_musique
from loqi.eval.runner import evaluate_retrieval, evaluate_trigger_scenarios
from loqi.graph.embeddings import EmbeddingModel
from loqi.pipeline.config import (
    FLAT_RAG,
    GRAPH_ONLY,
    LOQI_FULL,
    LOQI_NO_DIFFUSE,
    LOQI_NO_HEBBIAN,
    LOQI_NO_TRIGGERS,
)
from loqi.retrieval.flat_rag import FlatRAG
from loqi.retrieval.graph_rag import GraphRAG
from loqi.retrieval.trigger_rag import TriggerRAG


def build_retrieval_system(variant_name, model):
    """Build a retrieval system for MuSiQue evaluation.

    For multi-hop retrieval benchmarks, triggers are not applicable
    (the documents are per-query paragraphs, not standing instructions).
    The relevant ablation dimensions are: graph on/off, diffuse on/off,
    Hebbian on/off.
    """
    if variant_name == "flat-rag":
        return FlatRAG(embedding_model=model)
    elif variant_name == "graph-only":
        return GraphRAG(config=GRAPH_ONLY, embedding_model=model)
    elif variant_name == "loqi-no-triggers":
        return GraphRAG(config=LOQI_NO_TRIGGERS, embedding_model=model)
    elif variant_name == "loqi-no-diffuse":
        return GraphRAG(config=LOQI_NO_DIFFUSE, embedding_model=model)
    elif variant_name == "loqi-no-hebbian":
        return GraphRAG(config=LOQI_NO_HEBBIAN, embedding_model=model)
    elif variant_name == "loqi-full":
        return GraphRAG(config=LOQI_FULL, embedding_model=model)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")


def build_trigger_system(variant_name, model):
    """Build a system for the trigger benchmark.

    Only trigger-enabled variants get TriggerRAG wrapping.
    Non-trigger variants use their base retrieval system.
    """
    if variant_name == "flat-rag":
        return FlatRAG(embedding_model=model)
    elif variant_name == "graph-only":
        return GraphRAG(config=GRAPH_ONLY, embedding_model=model)
    elif variant_name == "loqi-no-triggers":
        return GraphRAG(config=LOQI_NO_TRIGGERS, embedding_model=model)
    elif variant_name == "loqi-no-diffuse":
        base = FlatRAG(embedding_model=model)
        return TriggerRAG(base_system=base, config=LOQI_NO_DIFFUSE, embedding_model=model)
    elif variant_name == "loqi-no-hebbian":
        base = FlatRAG(embedding_model=model)
        return TriggerRAG(base_system=base, config=LOQI_NO_HEBBIAN, embedding_model=model)
    elif variant_name == "loqi-full":
        base = FlatRAG(embedding_model=model)
        return TriggerRAG(base_system=base, config=LOQI_FULL, embedding_model=model)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")


def main():
    parser = argparse.ArgumentParser(description="Run Loqi ablation study")
    parser.add_argument("--n-examples", type=int, default=100,
                       help="Number of MuSiQue examples to use")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    model = EmbeddingModel()

    # Load benchmarks
    print("Loading benchmarks...")
    all_musique = load_musique(data_dir / "raw" / "musique" / "validation.jsonl")
    answerable = [e for e in all_musique if e.metadata.get("answerable", True)]
    musique_examples = answerable[:args.n_examples]
    print(f"  MuSiQue: {len(musique_examples)} answerable examples")

    trigger_scenarios = load_trigger_scenarios(data_dir / "custom_benchmark" / "scenarios")
    trigger_memories = load_memories(data_dir / "custom_benchmark" / "memories")
    print(f"  Trigger benchmark: {len(trigger_scenarios)} scenarios, {len(trigger_memories)} memories")

    variants = [
        "flat-rag",
        "graph-only",
        "loqi-no-triggers",
        "loqi-no-diffuse",
        "loqi-no-hebbian",
        "loqi-full",
    ]

    # --- MuSiQue ablation ---
    print(f"\n{'='*70}")
    print(f"MUSIQUE ABLATION (n={len(musique_examples)})")
    print(f"{'='*70}")

    musique_results = {}
    for variant in variants:
        print(f"\n  Running {variant}...", end=" ", flush=True)
        t0 = time.time()
        system = build_retrieval_system(variant, model)
        result = evaluate_retrieval(system, musique_examples, top_k=5)
        elapsed = time.time() - t0
        musique_results[variant] = result
        r5 = result.aggregate.get("recall@5", {}).get("mean", 0)
        sf1 = result.aggregate.get("support_f1", {}).get("mean", 0)
        print(f"recall@5={r5:.4f}  support_f1={sf1:.4f}  ({elapsed:.1f}s)")

    # Print comparison table
    print(f"\n{'Variant':<22} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'Sup_F1':>8}")
    print("-" * 56)
    for variant in variants:
        r = musique_results[variant]
        r1 = r.aggregate.get("recall@1", {}).get("mean", 0)
        r3 = r.aggregate.get("recall@3", {}).get("mean", 0)
        r5 = r.aggregate.get("recall@5", {}).get("mean", 0)
        sf = r.aggregate.get("support_f1", {}).get("mean", 0)
        print(f"{variant:<22} {r1:>8.4f} {r3:>8.4f} {r5:>8.4f} {sf:>8.4f}")

    # --- Trigger ablation ---
    print(f"\n{'='*70}")
    print(f"TRIGGER ABLATION ({len(trigger_scenarios)} scenarios)")
    print(f"{'='*70}")

    trigger_results = {}
    for variant in variants:
        print(f"\n  Running {variant}...", end=" ", flush=True)
        system = build_trigger_system(variant, model)
        result = evaluate_trigger_scenarios(system, trigger_scenarios, trigger_memories)
        trigger_results[variant] = result
        tr = result["aggregate"]["trigger_recall"]["mean"]
        tp = result["aggregate"]["trigger_precision"]["mean"]
        print(f"recall={tr:.4f}  precision={tp:.4f}")

    # Print comparison table
    print(f"\n{'Variant':<22} {'Trig_Recall':>12} {'Trig_Prec':>12}")
    print("-" * 48)
    for variant in variants:
        r = trigger_results[variant]
        tr = r["aggregate"]["trigger_recall"]["mean"]
        tp = r["aggregate"]["trigger_precision"]["mean"]
        print(f"{variant:<22} {tr:>12.4f} {tp:>12.4f}")

    # Save results
    output_dir = data_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_data = {
        "musique": {
            variant: {
                "aggregate": musique_results[variant].aggregate,
                "n_examples": len(musique_examples),
            }
            for variant in variants
        },
        "trigger": {
            variant: trigger_results[variant]["aggregate"]
            for variant in variants
        },
    }
    output_path = output_dir / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
