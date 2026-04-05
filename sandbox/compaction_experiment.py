"""Compaction experiment v2 -- policy memory that can't be inferred from code.

Tests whether Loqi preserves institutional knowledge (project policies,
deployment rules, database decisions) across context compaction events.

These policies are invisible in the codebase -- they exist only in
standing instructions. After compaction, a model without external
memory must guess. A model with Loqi gets the policies re-injected.

Usage: python sandbox/compaction_experiment.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from loqi.benchmarks.schema import Document
from loqi.graph.embeddings import EmbeddingModel
from loqi.graph.models import TriggerOrigin
from loqi.llm.client import OllamaClient
from loqi.pipeline.config import PipelineConfig
from loqi.retrieval.section_retrieval import SectionRetrieval


ACTING_MODEL = "qwen2.5-coder:14b"

WITH_MEMORY_PROMPT = """You are a senior engineer on this project. You know the project's standing policies and must follow them.

{memory_section}

## Question

{task_instruction}

## Instructions

Answer the question following all project policies above. Be specific about which policies apply and why. Keep your answer concise (3-5 sentences)."""

NO_MEMORY_PROMPT = """You are a senior engineer. Answer the following question about a software project.

## Question

{task_instruction}

## Instructions

Answer concisely (3-5 sentences). Give your best recommendation based on general best practices."""

MEMORY_HEADER = """## PROJECT POLICIES -- you MUST follow these

{sections}"""


def build_mem(sections):
    if not sections:
        return ""
    parts = [f"Policy: {s['title']}\n{s['content']}" for s in sections]
    return MEMORY_HEADER.format(sections="\n\n".join(parts))


def check_response(response, task):
    lower = response.lower()
    check_for = task.get("check_for", [])
    check_against = task.get("check_against", [])

    found = [kw for kw in check_for if kw.lower() in lower]
    violations = [kw for kw in check_against if kw.lower() in lower]

    recall = len(found) / len(check_for) if check_for else 1.0
    clean = len(violations) == 0
    score = recall * (1.0 if clean else 0.5)

    return {
        "recall": recall,
        "found": found,
        "missed": [kw for kw in check_for if kw.lower() not in lower],
        "violations": violations,
        "clean": clean,
        "score": score,
    }


def main():
    base_dir = Path(__file__).resolve().parent
    with open(base_dir / "tasks_compaction.yaml", encoding="utf-8") as f:
        all_tasks = yaml.safe_load(f)["tasks"]

    memories_dir = base_dir / "memories"
    ollama = OllamaClient(timeout=120)
    if not ollama.is_available():
        print("ERROR: Ollama not running")
        return

    model = EmbeddingModel()
    config = PipelineConfig(
        enable_graph=True, enable_triggers=True, enable_diffuse=True,
        enable_hebbian=True, enable_llm_gate=True,
        trigger_confidence_threshold=0.15, hebbian_strengthen_rate=0.15,
        hebbian_promotion_threshold_soft=2, hebbian_promotion_threshold_hard=4,
        hebbian_promotion_threshold_trigger=6,
    )

    mem_docs = []
    for md in sorted(memories_dir.glob("*.md")):
        mem_docs.append(Document(
            id=md.name, title=md.stem, text=md.read_text(encoding="utf-8"),
        ))

    loqi = SectionRetrieval(config=config, embedding_model=model)
    for doc in mem_docs:
        loqi.index([doc])

    compaction_at = 5
    pre_tasks = all_tasks[:compaction_at]
    post_tasks = all_tasks[compaction_at:]

    print("=" * 65)
    print("COMPACTION EXPERIMENT v2 -- Policy Memory")
    print("=" * 65)
    print(f"  {len(pre_tasks)} pre-compaction, {len(post_tasks)} post-compaction")

    static_sections = [{"title": s.title, "content": s.content[:400]}
                      for s in loqi._section_nodes]
    static_mem = build_mem(static_sections)

    all_results = {"with_loqi": [], "no_loqi": []}

    for run_label, use_loqi in [("WITH LOQI", True), ("WITHOUT LOQI", False)]:
        print(f"\n{'_' * 65}")
        print(f"  RUN: {run_label}")
        print(f"{'_' * 65}")

        results = []

        print("\n  Pre-compaction (policies in prompt):")
        for task in pre_tasks:
            prompt = WITH_MEMORY_PROMPT.format(
                memory_section=static_mem,
                task_instruction=task["instruction"],
            )

            print(f"    {task['id']}: {task['instruction'][:50]}...", end=" ", flush=True)
            t0 = time.time()
            response = ollama.generate(ACTING_MODEL, prompt)
            elapsed = time.time() - t0

            check = check_response(response, task)
            print(f"score={check['score']:.0%} ({elapsed:.1f}s)")

            if use_loqi:
                result = loqi.retrieve(task["instruction"], top_k=4)
                useful = set(d.id for d in result.retrieved_docs)
                loqi.update(task["instruction"], result, useful)

            results.append({
                "task": task["id"], "phase": "pre",
                "score": check["score"], "found": check["found"],
                "missed": check["missed"], "violations": check["violations"],
            })

        if use_loqi:
            loqi.consolidate()

        store = loqi._store
        triggers = store.get_all_triggers()
        hebbian = [t for t in triggers if t.origin == TriggerOrigin.HEBBIAN]
        print(f"\n  *** COMPACTION ***")
        if use_loqi:
            print(f"  Loqi: {store.get_node_count()} nodes, {store.get_edge_count()} edges, "
                  f"{len(triggers)} triggers ({len(hebbian)} Hebbian)")
        else:
            print(f"  No external memory available")

        print(f"\n  Post-compaction ({'Loqi re-injects' if use_loqi else 'no policies'}):")
        for task in post_tasks:
            if use_loqi:
                result = loqi.retrieve(task["instruction"], top_k=4)
                mem_secs = [{"title": d.title, "content": d.text[:400]}
                           for d in result.retrieved_docs]
                prompt = WITH_MEMORY_PROMPT.format(
                    memory_section=build_mem(mem_secs),
                    task_instruction=task["instruction"],
                )
            else:
                prompt = NO_MEMORY_PROMPT.format(
                    task_instruction=task["instruction"],
                )

            print(f"    {task['id']}: {task['instruction'][:50]}...", end=" ", flush=True)
            t0 = time.time()
            response = ollama.generate(ACTING_MODEL, prompt)
            elapsed = time.time() - t0

            check = check_response(response, task)
            print(f"score={check['score']:.0%} ({elapsed:.1f}s)")
            if check["missed"]:
                print(f"      missed: {check['missed']}")
            if check["violations"]:
                print(f"      violations: {check['violations']}")

            results.append({
                "task": task["id"], "phase": "post",
                "score": check["score"], "found": check["found"],
                "missed": check["missed"], "violations": check["violations"],
            })

        key = "with_loqi" if use_loqi else "no_loqi"
        all_results[key] = results

    # === RESULTS ===
    print(f"\n{'=' * 65}")
    print("RESULTS")
    print(f"{'=' * 65}")

    for phase_label, phase in [("Pre-compaction", "pre"), ("Post-compaction", "post")]:
        a = [r for r in all_results["with_loqi"] if r["phase"] == phase]
        b = [r for r in all_results["no_loqi"] if r["phase"] == phase]
        a_mean = sum(r["score"] for r in a) / len(a) if a else 0
        b_mean = sum(r["score"] for r in b) / len(b) if b else 0
        d = a_mean - b_mean
        print(f"\n  {phase_label}:")
        print(f"    With Loqi: {a_mean:.0%}   No Loqi: {b_mean:.0%}   Delta: {d:+.0%}")

    print(f"\n  Post-compaction detail:")
    post_a = [r for r in all_results["with_loqi"] if r["phase"] == "post"]
    post_b = [r for r in all_results["no_loqi"] if r["phase"] == "post"]
    print(f"  {'Task':<12} {'With Loqi':>10} {'No Loqi':>10} {'Delta':>8}")
    print(f"  {'-' * 42}")
    for ra, rb in zip(post_a, post_b):
        d = ra["score"] - rb["score"]
        m = "+" if d > 0 else " "
        print(f"  {ra['task']:<12} {ra['score']:>9.0%} {rb['score']:>9.0%} {m}{d:>6.0%}")

    pre_a_m = sum(r["score"] for r in all_results["with_loqi"] if r["phase"] == "pre") / compaction_at
    post_a_m = sum(r["score"] for r in all_results["with_loqi"] if r["phase"] == "post") / len(post_tasks)
    pre_b_m = sum(r["score"] for r in all_results["no_loqi"] if r["phase"] == "pre") / compaction_at
    post_b_m = sum(r["score"] for r in all_results["no_loqi"] if r["phase"] == "post") / len(post_tasks)

    print(f"\n  Compaction impact:")
    print(f"    With Loqi: {pre_a_m:.0%} -> {post_a_m:.0%} (drop: {post_a_m-pre_a_m:+.0%})")
    print(f"    No Loqi:   {pre_b_m:.0%} -> {post_b_m:.0%} (drop: {post_b_m-pre_b_m:+.0%})")

    output_dir = base_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "compaction_experiment.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to results/compaction_experiment.json")


if __name__ == "__main__":
    main()
