"""Hard temporal benchmark -- forces flat RAG to fail.

Uses a larger corpus with semantic confounds, long-horizon memory,
co-activation patterns, and trigger suppression tests.

Runs flat RAG vs Loqi v2 side by side with per-channel reporting.

Usage: python scripts/run_hard_benchmark.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from loqi.benchmarks.schema import Document
from loqi.graph.embeddings import EmbeddingModel
from loqi.graph.models import NodeType, TriggerOrigin
from loqi.pipeline.config import PipelineConfig
from loqi.retrieval.flat_rag import FlatRAG
from loqi.retrieval.section_retrieval import SectionRetrieval


def load_corpus(corpus_path):
    with open(corpus_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    corpus_path = data_dir / "temporal_benchmark" / "corpus.yaml"
    corpus = load_corpus(corpus_path)
    model = EmbeddingModel()

    config = PipelineConfig(
        enable_graph=True,
        enable_triggers=True,
        enable_diffuse=True,
        enable_hebbian=True,
        hebbian_strengthen_rate=0.15,
        hebbian_promotion_threshold_soft=2,
        hebbian_promotion_threshold_hard=4,
        hebbian_promotion_threshold_trigger=6,
        trigger_confidence_threshold=0.15,
    )

    loqi = SectionRetrieval(config=config, embedding_model=model)
    flat = FlatRAG(embedding_model=model)

    print("=" * 70)
    print("HARD TEMPORAL BENCHMARK")
    print("=" * 70)

    all_events = []
    for doc in corpus["documents"]:
        all_events.append(("doc", doc["day"], doc))
    for query in corpus["queries"]:
        all_events.append(("query", query["day"], query))
    all_events.sort(key=lambda x: (x[1], 0 if x[0] == "doc" else 1))

    results = []
    current_day = 0
    all_docs_for_flat = []

    for event_type, day, data in all_events:
        if day > current_day:
            if current_day > 0:
                report = loqi.consolidate()
                if report.edges_strengthened or report.bridges_created or report.trigger_candidates:
                    print(f"  [Consolidation day {current_day}] "
                          f"str={report.edges_strengthened} "
                          f"br={report.bridges_created} "
                          f"tr={report.trigger_candidates}")
            current_day = day

        if event_type == "doc":
            print(f"\n  Day {day}: INGEST {data['id']}")
            doc_obj = Document(id=data["id"], title=data["title"], text=data["content"])
            loqi.index([doc_obj])
            all_docs_for_flat.append(doc_obj)
            store = loqi._store
            sections = [n for n in store.get_all_nodes() if n.node_type == NodeType.SECTION]
            print(f"    Sections: {len(sections)}, Edges: {store.get_edge_count()}")

        elif event_type == "query":
            difficulty = data.get("difficulty", "unknown")
            expected = set(data.get("expected", []))

            # Training episodes: provide feedback but don't score
            if difficulty == "training":
                result_train = loqi.retrieve(data["context"], top_k=6)
                if expected:
                    loqi.update(data["context"], result_train, expected)
                print(f"\n  Day {day} [training]: \"{data['context'][:55]}...\" (feedback only)")
                continue

            print(f"\n  Day {day} [{difficulty}]: \"{data['context'][:55]}...\"")

            # Loqi v2
            result_loqi = loqi.retrieve(data["context"], top_k=6)
            surfaced_loqi = set(result_loqi.retrieved_ids[:6]) | result_loqi.triggered_memories
            hit_loqi = surfaced_loqi & expected

            if expected:
                recall_loqi = len(hit_loqi) / len(expected)
            else:
                # Trigger suppression: score 1.0 if no triggers fired, 0.0 if they did
                recall_loqi = 1.0 if not result_loqi.triggered_memories else 0.0

            # Hebbian feedback
            if expected:
                loqi.update(data["context"], result_loqi, expected)

            # Flat RAG baseline — section-level for fair comparison
            # Uses SectionRetrieval with triggers/graph/hebbian OFF
            # This is pure semantic search on sections (the true baseline)
            flat_config = PipelineConfig(
                enable_graph=False,
                enable_triggers=False,
                enable_diffuse=False,
                enable_hebbian=False,
            )
            flat_system = SectionRetrieval(config=flat_config, embedding_model=model)
            for doc_obj in all_docs_for_flat:
                flat_system.index([doc_obj])
            result_flat = flat_system.retrieve(data["context"], top_k=6)
            surfaced_flat = set(result_flat.retrieved_ids[:6])

            if expected:
                hit_flat = surfaced_flat & expected
                recall_flat = len(hit_flat) / len(expected)
            else:
                # Flat has no triggers, so suppression is trivially correct
                recall_flat = 1.0

            # Channel attribution
            meta = result_loqi.metadata
            semantic = set(meta.get("semantic_section_ids", []))
            triggered = set(meta.get("triggered_section_ids", []))
            graph_disc = set(meta.get("graph_discovered_section_ids", []))

            channel_wins = {}
            for exp_id in expected:
                if exp_id in triggered:
                    channel_wins[exp_id] = "trigger"
                elif exp_id in graph_disc:
                    channel_wins[exp_id] = "graph"
                elif exp_id in semantic:
                    channel_wins[exp_id] = "semantic"
                else:
                    channel_wins[exp_id] = "MISS"

            semantic_rescued = [
                eid for eid, ch in channel_wins.items()
                if ch in ("trigger", "graph") and eid not in semantic
            ]

            delta = recall_loqi - recall_flat
            marker = "+" if delta > 0 else ("-" if delta < 0 else " ")

            print(f"    Loqi: {recall_loqi:.2f}  Flat: {recall_flat:.2f}  [{marker}{delta:+.2f}]")
            if expected:
                for eid, ch in channel_wins.items():
                    print(f"      {eid}: {ch}")
            else:
                fired = result_loqi.triggered_memories
                if fired:
                    print(f"      FALSE TRIGGERS: {sorted(fired)}")
                else:
                    print(f"      Correctly suppressed")

            if semantic_rescued:
                print(f"      ** Rescued by graph/trigger: {semantic_rescued}")

            results.append({
                "day": day,
                "difficulty": difficulty,
                "context": data["context"][:80],
                "expected": sorted(expected),
                "recall_loqi": recall_loqi,
                "recall_flat": recall_flat,
                "delta": delta,
                "channel_wins": channel_wins,
                "semantic_rescued": semantic_rescued,
                "triggered": sorted(result_loqi.triggered_memories),
            })

    report = loqi.consolidate()
    print(f"\n  [Final] {report.summary()}")

    # === RESULTS ===
    print(f"\n{'=' * 70}")
    print("RESULTS BY DIFFICULTY")
    print(f"{'=' * 70}")

    difficulties = ["easy", "semantic_confound", "long_horizon",
                     "co_activation", "trigger_suppression"]
    for diff in difficulties:
        subset = [r for r in results if r["difficulty"] == diff]
        if not subset:
            continue
        loqi_m = sum(r["recall_loqi"] for r in subset) / len(subset)
        flat_m = sum(r["recall_flat"] for r in subset) / len(subset)
        d = loqi_m - flat_m
        rescued = sum(len(r["semantic_rescued"]) for r in subset)
        print(f"\n  {diff} (n={len(subset)})")
        print(f"    Loqi v2: {loqi_m:.3f}  Flat RAG: {flat_m:.3f}  Delta: {d:+.3f}")
        if rescued:
            print(f"    Sections rescued by graph/trigger: {rescued}")

    loqi_all = sum(r["recall_loqi"] for r in results) / len(results)
    flat_all = sum(r["recall_flat"] for r in results) / len(results)
    rescued_all = sum(len(r["semantic_rescued"]) for r in results)

    print(f"\n  OVERALL (n={len(results)})")
    print(f"    Loqi v2: {loqi_all:.3f}  Flat RAG: {flat_all:.3f}  Delta: {loqi_all - flat_all:+.3f}")
    print(f"    Sections rescued by graph/trigger: {rescued_all}")

    store = loqi._store
    all_triggers = store.get_all_triggers()
    hebbian = [t for t in all_triggers if t.origin == TriggerOrigin.HEBBIAN]
    print(f"\n  Graph: {store.get_node_count()} nodes, {store.get_edge_count()} edges")
    print(f"  Triggers: {len(all_triggers)} total, {len(hebbian)} Hebbian")

    output_dir = data_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "hard_benchmark.json", "w") as f:
        json.dump({
            "summary": {
                "loqi_overall": loqi_all,
                "flat_overall": flat_all,
                "delta": loqi_all - flat_all,
                "rescued": rescued_all,
            },
            "per_difficulty": {
                diff: {
                    "loqi": sum(r["recall_loqi"] for r in subset) / len(subset),
                    "flat": sum(r["recall_flat"] for r in subset) / len(subset),
                    "n": len(subset),
                }
                for diff in difficulties
                for subset in [[r for r in results if r["difficulty"] == diff]]
                if subset
            },
            "per_query": results,
        }, f, indent=2)
    print(f"\n  Saved to {output_dir / 'hard_benchmark.json'}")


if __name__ == "__main__":
    main()
