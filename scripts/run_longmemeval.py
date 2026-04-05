"""Run LongMemEval preference questions against Loqi's trigger system.

Tests whether Loqi's triggers can surface user preferences that were
mentioned in a past conversation session, when the user later asks
a generic question that doesn't mention the preference.

Example: User said "I use Adobe Premiere Pro" in session 12.
Later asks "recommend video editing resources." The system should
surface the Premiere Pro preference without being asked.

Two evaluation modes:
  Oracle: index only the answer session (tests trigger extraction + matching)
  Haystack: index all ~50 sessions (tests trigger recall from noise)

Usage: python scripts/run_longmemeval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loqi.benchmarks.loaders import load_longmemeval
from loqi.eval.metrics import retrieval_recall_at_k
from loqi.graph.embeddings import EmbeddingModel
from loqi.pipeline.config import PipelineConfig
from loqi.retrieval.flat_rag import FlatRAG
from loqi.retrieval.graph_rag import GraphRAG
from loqi.retrieval.trigger_rag import TriggerRAG


def run_preference_eval(examples, system, top_k=5):
    """Run preference evaluation and return per-example results."""
    results = []

    for ex in examples:
        system.reset()
        system.index(ex.documents)

        result = system.retrieve(ex.query, top_k=top_k)

        gold = set(ex.supporting_doc_ids)
        retrieved = set(result.retrieved_ids[:top_k])
        triggered = result.triggered_memories

        recall = retrieval_recall_at_k(result.retrieved_ids, gold, top_k)
        hit = bool(gold & (retrieved | triggered))

        results.append({
            "id": ex.id,
            "query": ex.query[:80],
            "answer_preview": ex.answer[:100],
            "n_docs": len(ex.documents),
            "gold": sorted(gold),
            "retrieved_top3": result.retrieved_ids[:3],
            "triggered": sorted(triggered),
            "recall": recall,
            "hit": hit,
        })

    return results


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    model = EmbeddingModel()

    config = PipelineConfig(
        enable_graph=True,
        enable_triggers=True,
        enable_diffuse=False,
        enable_hebbian=False,
        trigger_confidence_threshold=0.10,
    )

    # Load both versions
    oracle = load_longmemeval(data_dir / "raw" / "longmemeval" / "longmemeval_oracle.json")
    small = load_longmemeval(data_dir / "raw" / "longmemeval" / "longmemeval_s_cleaned.json")

    prefs_oracle = [e for e in oracle if e.category == "single-session-preference"]
    prefs_small = [e for e in small if e.category == "single-session-preference"]

    print(f"LongMemEval preference questions: {len(prefs_oracle)}")

    # === ORACLE MODE: 1 doc per question ===
    print(f"\n{'='*60}")
    print("ORACLE MODE (1 supporting session per question)")
    print(f"{'='*60}")

    flat = FlatRAG(embedding_model=model)
    flat_results = run_preference_eval(prefs_oracle, flat)
    flat_hit = sum(1 for r in flat_results if r["hit"]) / len(flat_results)
    flat_recall = sum(r["recall"] for r in flat_results) / len(flat_results)
    print(f"\n  Flat RAG: hit_rate={flat_hit:.3f}, recall@5={flat_recall:.3f}")

    base = GraphRAG(config=config, embedding_model=model)
    trig_sys = TriggerRAG(base_system=base, config=config, embedding_model=model)
    trig_results = run_preference_eval(prefs_oracle, trig_sys)
    trig_hit = sum(1 for r in trig_results if r["hit"]) / len(trig_results)
    trig_recall = sum(r["recall"] for r in trig_results) / len(trig_results)
    print(f"  TriggerRAG: hit_rate={trig_hit:.3f}, recall@5={trig_recall:.3f}")

    # === HAYSTACK MODE: ~50 docs per question ===
    print(f"\n{'='*60}")
    print("HAYSTACK MODE (~50 sessions per question)")
    print(f"{'='*60}")

    flat2 = FlatRAG(embedding_model=model)
    flat_results_h = run_preference_eval(prefs_small, flat2)
    flat_hit_h = sum(1 for r in flat_results_h if r["hit"]) / len(flat_results_h)
    flat_recall_h = sum(r["recall"] for r in flat_results_h) / len(flat_results_h)
    print(f"\n  Flat RAG: hit_rate={flat_hit_h:.3f}, recall@5={flat_recall_h:.3f}")

    base2 = GraphRAG(config=config, embedding_model=model)
    trig_sys2 = TriggerRAG(base_system=base2, config=config, embedding_model=model)
    trig_results_h = run_preference_eval(prefs_small, trig_sys2)
    trig_hit_h = sum(1 for r in trig_results_h if r["hit"]) / len(trig_results_h)
    trig_recall_h = sum(r["recall"] for r in trig_results_h) / len(trig_results_h)
    print(f"  TriggerRAG: hit_rate={trig_hit_h:.3f}, recall@5={trig_recall_h:.3f}")

    # === COMPARISON ===
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    print(f"\n{'Mode':<15} {'System':<15} {'Hit Rate':>10} {'Recall@5':>10}")
    print("-" * 52)
    print(f"{'Oracle':<15} {'Flat RAG':<15} {flat_hit:>10.3f} {flat_recall:>10.3f}")
    print(f"{'Oracle':<15} {'TriggerRAG':<15} {trig_hit:>10.3f} {trig_recall:>10.3f}")
    print(f"{'Haystack':<15} {'Flat RAG':<15} {flat_hit_h:>10.3f} {flat_recall_h:>10.3f}")
    print(f"{'Haystack':<15} {'TriggerRAG':<15} {trig_hit_h:>10.3f} {trig_recall_h:>10.3f}")

    # Per-query detail for haystack
    print(f"\n{'='*60}")
    print("HAYSTACK PER-QUERY DETAIL (differences only)")
    print(f"{'='*60}")

    improved = 0
    regressed = 0
    for fr, tr in zip(flat_results_h, trig_results_h):
        delta = tr["recall"] - fr["recall"]
        if delta > 0:
            improved += 1
        elif delta < 0:
            regressed += 1

        if delta != 0:
            marker = "+" if delta > 0 else "-"
            print(f"\n  {marker} {tr['id']}: Flat={fr['recall']:.2f} Trigger={tr['recall']:.2f}")
            print(f"    Query: {tr['query']}")
            if tr["triggered"]:
                print(f"    Triggered: {tr['triggered'][:3]}")

    same = len(flat_results_h) - improved - regressed
    print(f"\n  Improved: {improved}, Regressed: {regressed}, Same: {same}")

    # Save results
    output_dir = data_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "longmemeval_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "oracle": {
                "flat_rag": {"hit_rate": flat_hit, "recall": flat_recall},
                "trigger_rag": {"hit_rate": trig_hit, "recall": trig_recall},
            },
            "haystack": {
                "flat_rag": {"hit_rate": flat_hit_h, "recall": flat_recall_h},
                "trigger_rag": {"hit_rate": trig_hit_h, "recall": trig_recall_h},
                "per_query": [
                    {
                        "id": tr["id"],
                        "query": tr["query"],
                        "n_docs": tr["n_docs"],
                        "flat_recall": fr["recall"],
                        "trigger_recall": tr["recall"],
                        "flat_hit": fr["hit"],
                        "trigger_hit": tr["hit"],
                        "triggered": tr["triggered"],
                    }
                    for fr, tr in zip(flat_results_h, trig_results_h)
                ],
            },
            "n_questions": len(prefs_oracle),
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
