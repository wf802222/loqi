"""Minimal local agent loop with Loqi memory.

Runs a sequence of coding tasks through a local LLM (qwen2.5-coder)
with Loqi providing pre-task memory retrieval and post-task learning.

Three modes for comparison:
  1. no_memory:     agent gets only the task instruction
  2. flat_retrieval: agent gets top-k similar sections (no triggers/graph)
  3. loqi_memory:    agent gets full retrieval (triggers + graph + learning)

Usage: python sandbox/agent_loop.py [--mode loqi_memory]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from loqi.benchmarks.schema import Document
from loqi.graph.embeddings import EmbeddingModel
from loqi.graph.models import NodeType, TriggerOrigin
from loqi.llm.client import OllamaClient
from loqi.pipeline.config import PipelineConfig
from loqi.retrieval.section_retrieval import SectionRetrieval


ACTING_MODEL = "qwen2.5-coder:14b"

AGENT_PROMPT = """You are a code assistant. You will be given a task to perform on a Python file.

{memory_section}

## Current file: app.py

```python
{file_content}
```

## Task

{task_instruction}

## Instructions

Output ONLY the corrected/updated Python code for app.py.
Do not include explanations, markdown formatting, or anything else.
Just output the raw Python code."""

MEMORY_HEADER = """## Project Memory

The following standing instructions and prior learnings are relevant to this task:

{sections}

Apply these rules when completing the task."""


def load_tasks(tasks_path):
    with open(tasks_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["tasks"]


def load_memories(memories_dir):
    docs = []
    for md_file in sorted(memories_dir.glob("*.md")):
        docs.append(Document(
            id=md_file.name,
            title=md_file.stem,
            text=md_file.read_text(encoding="utf-8"),
        ))
    return docs


def build_memory_section(sections):
    if not sections:
        return ""
    parts = []
    for s in sections:
        parts.append(f"### {s['title']}\n{s['content']}")
    section_text = "\n\n".join(parts)
    return MEMORY_HEADER.format(sections=section_text)


def run_agent_loop(mode):
    base_dir = Path(__file__).resolve().parent
    tasks = load_tasks(base_dir / "tasks.yaml")
    memories = load_memories(base_dir / "memories")
    project_file = base_dir / "project" / "app.py"

    ollama = OllamaClient(timeout=120)
    if not ollama.is_available():
        print("ERROR: Ollama is not running. Start it first.")
        return

    model = EmbeddingModel()

    # Build Loqi system based on mode
    if mode == "loqi_memory":
        config = PipelineConfig(
            enable_graph=True,
            enable_triggers=True,
            enable_diffuse=True,
            enable_hebbian=True,
            enable_llm_gate=True,
            trigger_confidence_threshold=0.15,
            hebbian_strengthen_rate=0.15,
            hebbian_promotion_threshold_soft=2,
            hebbian_promotion_threshold_hard=4,
            hebbian_promotion_threshold_trigger=6,
        )
    elif mode == "flat_retrieval":
        config = PipelineConfig(
            enable_graph=False,
            enable_triggers=False,
            enable_diffuse=False,
            enable_hebbian=False,
        )
    else:
        config = None  # no memory

    loqi = None
    if config:
        loqi = SectionRetrieval(config=config, embedding_model=model)
        for doc in memories:
            loqi.index([doc])

    print("=" * 60)
    print(f"AGENT LOOP -- mode: {mode}")
    print("=" * 60)
    print(f"Acting model: {ACTING_MODEL}")
    print(f"Tasks: {len(tasks)}")
    print(f"Memories indexed: {len(memories)} documents")
    if loqi:
        store = loqi._store
        sections = [n for n in store.get_all_nodes() if n.node_type == NodeType.SECTION]
        print(f"Sections: {len(sections)}, Edges: {store.get_edge_count()}")

    results = []

    for i, task in enumerate(tasks):
        print(f"\n{'_' * 60}")
        print(f"Task {i+1}/{len(tasks)}: {task['id']}")
        print(f"  {task['instruction'][:70]}...")

        # Read current file state
        file_content = project_file.read_text(encoding="utf-8")

        # Retrieve memory
        memory_sections = []
        triggered = set()
        result = None
        if loqi:
            result = loqi.retrieve(task["instruction"], top_k=4)
            triggered = result.triggered_memories
            for doc in result.retrieved_docs:
                memory_sections.append({
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.text[:300],
                })
            meta = result.metadata
            print(f"  Memory: {len(memory_sections)} sections retrieved")
            print(f"    triggered: {sorted(triggered)}")

            # Check expected memory
            expected = set(task.get("expected_memory", []))
            surfaced = set(r["id"] for r in memory_sections) | triggered
            hit = surfaced & expected
            recall = len(hit) / len(expected) if expected else 1.0
            print(f"    expected memory recall: {recall:.2f}")
        else:
            print(f"  Memory: (none -- {mode} mode)")

        # Build prompt
        mem_text = build_memory_section(memory_sections)
        prompt = AGENT_PROMPT.format(
            memory_section=mem_text,
            file_content=file_content,
            task_instruction=task["instruction"],
        )

        # Call acting model
        print(f"  Calling {ACTING_MODEL}...", end=" ", flush=True)
        t0 = time.time()
        response = ollama.generate(ACTING_MODEL, prompt)
        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s)")

        # Extract code from response
        code = response.strip()
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        # Check expected changes
        expected_changes = task.get("expected_changes", [])
        changes_found = []
        for change in expected_changes:
            if change.lower() in code.lower():
                changes_found.append(change)

        change_recall = (
            len(changes_found) / len(expected_changes) if expected_changes else 1.0
        )
        print(f"  Expected changes: {len(changes_found)}/{len(expected_changes)} ({change_recall:.0%})")
        for c in expected_changes:
            status = "found" if c in changes_found else "MISSING"
            print(f"    {c}: {status}")

        # Write updated file
        if code and len(code) > 50:
            project_file.write_text(code, encoding="utf-8")
            print(f"  File updated ({len(code)} chars)")
        else:
            print(f"  WARNING: response too short ({len(code)} chars), skipping write")

        # Hebbian feedback
        if loqi and result and memory_sections:
            useful = set()
            for ms in memory_sections:
                for change in changes_found:
                    if change.lower() in ms["content"].lower():
                        useful.add(ms["id"])
            if useful:
                loqi.update(task["instruction"], result, useful)
                print(f"  Hebbian feedback: {sorted(useful)}")

        # Consolidation between tasks
        if loqi and i < len(tasks) - 1:
            report = loqi.consolidate()
            if report.edges_strengthened or report.trigger_candidates:
                print(f"  Consolidation: str={report.edges_strengthened} "
                      f"tr={report.trigger_candidates}")

        results.append({
            "task_id": task["id"],
            "mode": mode,
            "memory_sections": [s["id"] for s in memory_sections],
            "triggered": sorted(triggered),
            "change_recall": change_recall,
            "changes_found": changes_found,
            "latency_s": elapsed,
        })

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY -- {mode}")
    print("=" * 60)
    mean_change = sum(r["change_recall"] for r in results) / len(results)
    print(f"  Mean change recall: {mean_change:.2f}")
    for r in results:
        print(
            f"  {r['task_id']}: {r['change_recall']:.0%} "
            f"({len(r['memory_sections'])} sections, {r['latency_s']:.1f}s)"
        )

    if loqi:
        store = loqi._store
        all_triggers = store.get_all_triggers()
        hebbian = [t for t in all_triggers if t.origin == TriggerOrigin.HEBBIAN]
        print(f"\n  Final graph: {store.get_node_count()} nodes, {store.get_edge_count()} edges")
        print(f"  Triggers: {len(all_triggers)} total, {len(hebbian)} Hebbian")

    # Save results
    output_dir = base_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / f"agent_loop_{mode}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to results/agent_loop_{mode}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent loop with Loqi memory")
    parser.add_argument(
        "--mode",
        choices=["no_memory", "flat_retrieval", "loqi_memory"],
        default="loqi_memory",
    )
    args = parser.parse_args()
    run_agent_loop(args.mode)
