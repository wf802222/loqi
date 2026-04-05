# /wrap — End of session wrap-up

Run at the end of every working session.

## Step 1 — Read current state
- `SESSIONS.md` (most recent entry for format reference)
- `DECISIONS.md` (check if any ADRs were added this session)
- `ROADMAP.md` (check if any phase status changed)

## Step 2 — Review the session
Identify:
- What was accomplished (files created/edited, bugs fixed, features added, tests run)
- What was decided (ADR numbers and one-line summaries)
- The single most important open blocker
- The recommended first action for next session

## Step 3 — Append to SESSIONS.md
Add a new entry at the TOP, just below `## Session Log (newest first)`:

---

### {YYYY-MM-DD} — {short title}

**Accomplished:**
- [bullet list]

**Decided:**
- [ADR numbers and summaries, or "none"]

**Open blocker:**
[single sentence]

**Recommended next session:**
[1-2 sentences]

---

## Step 4 — Update ROADMAP.md if needed
Surgical edits only — phase completions, resolved decisions, status changes.

## Step 5 — Confirm
Print:
- "Session logged in SESSIONS.md"
- "Open blocker: [one sentence]"
- "Next session: [one sentence]"
