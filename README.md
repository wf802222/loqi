# Loqi

*Memory that speaks when it matters.*

Loqi is an experimental memory architecture for AI systems. Instead of treating memory as a static document store queried only after the fact, Loqi is built around a different idea:

- memory is formed as work happens
- useful connections can be built before a later query arrives
- repeated useful co-activation should make future retrieval easier

Today, Loqi is best understood as a `continuous memory formation` prototype with:

- section-level memory objects
- write-time memory linking
- downtime consolidation / "dreaming"
- query-time retrieval through semantic, trigger, and graph channels
- a small local LLM gate for trigger suppression

## Current State

Loqi is not a production system. It is a research prototype with a working architecture v2.

What is implemented:

- `MemoryWriter` for write-time section creation and linking
- `Consolidator` for decay, replay, promotion, bridge discovery, and trigger mining
- `SectionRetrieval` for three-channel retrieval:
  - semantic similarity
  - trigger firing
  - graph traversal

What is currently interesting:

- Loqi now works at the `section` level rather than treating whole files as single memory units.
- It forms useful structure before later queries depend on it.
- It has early evidence of real advantage over flat retrieval in `semantic-confound` cases.
- It can improve with additional prior work episodes through Hebbian-style learning.
- A small local LLM layer can improve trigger suppression without replacing the deterministic memory core.

What is not yet proven:

- consistent overall superiority to strong flat RAG baselines
- robust proactive resurfacing in realistic environments
- large-scale long-horizon behavior
- institutional deployment readiness

## Core Idea

Loqi is built around three phases:

### 1. Write-Time Processing

When a document arrives, Loqi:
- splits it into section-level memory objects
- creates document and section nodes
- builds containment and cross-section edges
- extracts candidate triggers

### 2. Downtime Consolidation

Between work episodes, Loqi:
- decays weak or stale edges
- replays recently useful episodes
- promotes stronger connections
- discovers bridge edges
- mines trigger candidates

### 3. Query-Time Orchestration

When a query arrives, Loqi retrieves through:
- semantic similarity
- triggers
- graph traversal

Those signals are merged and ranked rather than collapsed into a single retrieval path.

## Why This Repo Exists

This repo is trying to answer a specific question:

`Can a memory system become more useful over time by living alongside work, rather than only retrieving from a frozen corpus?`

The strongest current signal is not that Loqi is universally better than flat RAG.

The strongest current signal is that Loqi begins to help in the kinds of cases where plain semantic retrieval gets confused, and that repeated useful work episodes can improve later retrieval.

## Repository Layout

- `src/loqi/graph/` - graph, writer, nodes, store
- `src/loqi/hebbian/` - episode log, updater, promoter, consolidator, decay
- `src/loqi/retrieval/` - section retrieval and retrieval policies
- `src/loqi/triggers/` - trigger extraction and matching
- `scripts/` - benchmark and experiment runners
- `tests/` - unit and integration tests
- `plans/` - architecture notes, findings, and next-step memos
- `results/` - saved experiment outputs

## Quickstart

### Install

With `uv`:

```bash
uv venv
uv pip install -e ".[dev,benchmarks]"
```

With `pip`:

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -e ".[dev,benchmarks]"
```

### Run Tests

```bash
pytest
```

### Main Docs

- [BENCHMARKS.md](./BENCHMARKS.md)
- [ROADMAP.md](./ROADMAP.md)
- [GPT.md](./GPT.md)
- [plans/architecture_v2.md](./plans/architecture_v2.md)
- [plans/llm_roles_v2_5.md](./plans/llm_roles_v2_5.md)
- [plans/evaluation_metric_hierarchy.md](./plans/evaluation_metric_hierarchy.md)

## Recommended Positioning

The honest positioning for this repo today is:

- research prototype
- memory architecture experiment
- evaluation harness for temporal memory systems

Not:

- production memory platform
- enterprise knowledge system
- proven replacement for strong baseline retrieval systems

## Short Version

Loqi is an early but real memory system that:

- stores work as section-level memory
- forms links as new material arrives
- consolidates those links during downtime
- retrieves through semantic, trigger, and graph channels
- is beginning to show real value in ambiguous retrieval settings

That is enough to make the repo worth shipping today.
