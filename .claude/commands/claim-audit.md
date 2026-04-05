# /claim-audit — Verify every thesis claim has backing evidence

Check that every claim in the thesis assessment is tied to an actual result artifact. Flag orphaned claims (stated but not backed) and stale claims (backed by old results).

## Step 1 — Read current claims

Read `plans/thesis_assessment.md` and extract every claim that uses specific numbers (percentages, deltas, counts).

## Step 2 — Check each claim against result artifacts

For each claim, verify:

1. **A result file exists** — the claim references data in `results/` (ablation_results.json, persistent_benchmark.json, longmemeval_results.json, etc.)
2. **The numbers match** — the claim's numbers match what's in the result file. Flag any mismatch.
3. **A script exists to reproduce it** — there's a script in `scripts/` that generates the result file.
4. **An integration test exists** (for mechanism claims) — claims about "the loop works" should reference a passing test in `tests/`.

## Step 3 — Report

Print a table:

```
CLAIM AUDIT — {date}
═══════════════════════════════════════════════════════

Claim                              Source File              Match
─────────────────────────────────────────────────────────────────
Trigger recall 66.7%               ablation_results.json    MATCH
LongMemEval +3.3pp                 longmemeval_results.json MATCH
Persistent benchmark +6.25pp       persistent_benchmark.json MATCH
Closed loop proven                 test_closed_loop.py      PASS
...

ORPHANED CLAIMS (stated but no backing artifact):
  - [list any claims without result files]

STALE CLAIMS (numbers don't match current results):
  - [list any claims where stated numbers != result file numbers]

VERDICT: {ALL CLAIMS BACKED / N CLAIMS NEED ATTENTION}
```

## Step 4 — Fix or flag

If any claims are orphaned or stale:
- Print the specific mismatch
- Suggest whether to update the claim or re-run the benchmark
- Do NOT auto-fix — present the finding and let the user decide

## When to run

Run /claim-audit:
- Before updating thesis_assessment.md
- Before writing any paper-facing text
- After re-running benchmarks (to check for result changes)
