# /run-evidence — Refresh the full evidence scorecard

Run all core benchmarks and print the current state of evidence for every thesis claim.

## What to run

Execute the following in sequence, capturing results:

1. **Custom trigger benchmark** — `python scripts/run_ablation.py --n-examples 50`
   Extract: trigger recall and precision for flat-rag vs trigger-enabled variants

2. **Persistent-corpus benchmark** — `python scripts/run_persistent_benchmark.py`
   Extract: trained vs untrained recall, delta, Hebbian triggers created

3. **LongMemEval preferences** — `python scripts/run_longmemeval.py`
   Extract: haystack mode hit rate for flat RAG vs TriggerRAG

## How to report

Print a scorecard in this exact format:

```
═══════════════════════════════════════════════════════
LOQI EVIDENCE SCORECARD — {date}
═══════════════════════════════════════════════════════

TRIGGERS (custom benchmark)
  Without triggers: {recall}  With triggers: {recall}
  Delta: {delta}pp  Precision: {precision}
  Status: {SUPPORTED / WEAK / NOT SHOWN}

TRIGGERS (LongMemEval, ICLR 2025)
  Flat RAG: {hit_rate}  TriggerRAG: {hit_rate}
  Delta: {delta}pp  n={n}
  Status: {SUPPORTED / WEAK / NOT SHOWN}

GRAPH RETRIEVAL (MuSiQue)
  Flat RAG R@5: {recall}  Graph R@5: {recall}
  Delta: {delta}pp  n={n}
  Status: {SUPPORTED / WEAK / NOT SHOWN}

HEBBIAN CLOSED LOOP (persistent corpus)
  Trained: {recall}  Untrained: {recall}
  Delta: {delta}pp  Learned-pair delta: {delta}pp
  Hebbian triggers created: {n}
  Status: {SUPPORTED / WEAK / NOT SHOWN}

CLOSED LOOP MECHANISM
  Integration test: {PASS / FAIL}
  Status: {PROVEN / NOT PROVEN}

═══════════════════════════════════════════════════════
```

Status rules:
- SUPPORTED: delta > 3pp AND statistically meaningful sample
- WEAK: delta > 0 but small sample or small margin
- NOT SHOWN: delta <= 0 or not tested
- PROVEN: integration test passes

## After printing

Save the scorecard to `results/scorecard.txt` (overwrite previous).

Print: "Evidence refreshed. Read results/scorecard.txt for the full state."
