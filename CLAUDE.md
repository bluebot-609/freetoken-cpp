# CLAUDE.md

This file is read automatically at the start of every Claude Code session in this repository. It explains what the two reference documents are, how to use them, and how to manage continuity across sessions.

---

## Reference Documents in This Repo

All three docs below live in `docs/` (not the repo root) — `CLAUDE.md` and `README.md` are the only project docs kept at root, `CLAUDE.md` because Claude Code auto-reads it specifically from there, `README.md` for GitHub's auto-render convention.

### `docs/roadmap.md`
The learning/planning document. Contains:
- Skill map (which parts of this project map to which skills the author is building — embedded systems background, C++, low-latency programming, GPU/CUDA, Linux internals, AI inference, agent orchestration)
- FreeToken (arXiv:2608.16157) as the explicit source of truth this project ports and extends — read Section 0 before touching scheduler code
- Phased roadmap with exit criteria per phase
- Novelty summary distinguishing what's ported vs. genuinely original (the hardware calibration pass is the headline original contribution)
- Resource list (books, papers, repos) per skill area

**When to consult it:** at the start of a new phase, when unsure why a design decision was made, or when the author asks "why are we doing it this way."

### `docs/dev_spec.md`
The engineering specification. Contains:
- Full repo structure
- Environment/toolkit choices and why
- Per-module responsibilities, exposed parameters, and required citations
- Benchmarking application design (comparison modes, scenario configs, output format)
- Testing strategy per level (unit/integration/regression/concurrency)
- Development workflow with profiling checkpoints baked into specific steps
- Explicit non-goals for v1

**When to consult it:** before implementing any module — it defines the module's interface, exposed parameters, and what must be cited. Treat Section 9 (Non-Goals) as a hard boundary; if a task seems to require something listed there, stop and flag it to the author rather than proceeding.

---

## Core Working Principles (apply to every session, not just the first)

1. **Lightweight over feature-complete.** Do not add abstraction layers "for future flexibility" unless a second concrete use case already demands them.
2. **Correctness of the q\* scheduler and calibration pass over breadth of features.** These are the actual point of the project. Everything else (server, harness, SSD tier) is secondary.
3. **Cite sources for every non-trivial design decision** — inline code comments plus an entry in `docs/citations.md` (format: `[date] [module] [source] — [what was learned/used]`). This is a stated learning requirement for the author, not optional polish. When you implement a module, end your summary with a short "what I referenced and why" note.
4. **Do not silently simplify the FreeToken design.** If you deviate from the paper's actual formulation (e.g., using a threshold heuristic instead of the closed-form optimal split), say so explicitly in comments and in your session summary — this is a described, intentional simplification, not a shortcut to hide.
5. **Follow `docs/dev_spec.md` Section 8's build order.** Baseline before scheduler, benchmarking tool alongside core modules (not after), profiling checkpoints at the specified steps — do not skip ahead to the server or harness before Phase 1's core mechanism is benchmarked and solid.

---

## How Code Gets Written — Teaching Mode (author is new to this stack)

The author is upskilling from embedded systems into C++/CUDA/ggml/Linux internals. Dumping a complete file leaves them unable to follow what happened. For **every** new module or non-trivial change, in this order:

1. **What and why, before any code.** One short paragraph: what this file/module does, why it's needed now (tie it to the specific `docs/dev_spec.md`/`docs/roadmap.md` step), and how it fits with what already exists. Wait for the author to be following before writing code — don't treat this as a formality to skip through.
2. **Build it in small chunks, not whole files.** Introduce one piece at a time (e.g., "first the struct/interface," then "now the function that fills it in," then "now wiring it into CMake") — each chunk gets a sentence or two on what it does and why it's shaped that way, not just a code block. A chunk should be small enough to actually read and understand in one sitting — a screenful, not a whole file.
3. **Check in, don't just plow ahead.** After a chunk that introduces something genuinely new (a new ggml API, a new concurrency primitive, a new pattern), pause rather than immediately writing the next five chunks — give the author a chance to ask "wait, why does it work like that" before context piles up.
4. **Still fine to move fast on:** mechanical/repetitive parts once the pattern's been shown once (e.g., a second nearly-identical CMakeLists.txt), and small fixes/corrections to something just written.

This applies for the whole project, not just the current module — re-read this section if it's been a while since it was followed.

---

## Session Continuity — Read This Before Doing Anything Else

Claude Code's built-in session resume does not reliably restore full prior context, especially across longer or multi-day sessions. **Do not assume you remember prior sessions.** Instead:

### On starting any session:
1. Read `docs/PROGRESS.md` (create it if it doesn't exist yet — see template below).
2. Read the most recent entries first — they reflect the current actual state, which may differ from what an earlier roadmap phase assumed.
3. State your understanding of current project state back to the author in one short paragraph before starting work, so any mismatch between `docs/PROGRESS.md` and reality gets caught immediately.
4. Check `git log --oneline -15` to cross-reference recent commits against what `docs/PROGRESS.md` claims — if they disagree, trust the git history and flag the discrepancy.

### During a session:
- Keep the built-in todo list (`/todos`) current for in-session task tracking. This is separate from `docs/PROGRESS.md` — todos are for the current session only.
- If a task or requirement seems to point outside the scope defined in `docs/dev_spec.md` Section 9 (Non-Goals), stop and ask rather than proceeding.

### Before ending any session (do this even if the author doesn't ask):
Update `docs/PROGRESS.md` with:
- What was completed this session (reference module numbers from `docs/dev_spec.md`, e.g. "Module 3.3")
- What's currently in-progress and in what state (e.g. "GPU-Expert Cache eviction logic written, unit tests not yet passing — failing on X")
- Anything that broke or is blocked, and why
- The concrete next step — specific enough that a fresh session could pick it up without re-deriving context
- Any new entries needed in `docs/citations.md`

Commit your work with messages tied to module numbers (e.g. `"Module 3.4: q* scheduler threshold split, benchmarked vs naive baseline"`) so git history itself is a secondary continuity record.

### `docs/PROGRESS.md` template (create this file if absent):
```markdown
# Progress Log

## Current Phase
[e.g. Phase 1 — GPU-Expert Cache + Host-Resident Pool]

## Last Updated
[date, session summary one-liner]

## Completed
- [Module X.Y]: [what, and current test/benchmark status]

## In Progress
- [Module X.Y]: [current state, what's blocking or half-done]

## Next Concrete Step
[specific enough for a fresh session to act on immediately]

## Open Questions / Flags for the Author
- [anything ambiguous, any Non-Goals boundary that was almost crossed, any deviation from FreeToken's design that needs sign-off]
```

---

## Quick Reference — Where Things Live
- Design rationale / "why" → `docs/roadmap.md`
- Module interfaces, parameters, citations required → `docs/dev_spec.md`
- Current actual state / what to do next → `docs/PROGRESS.md` (you maintain this)
- Sources consulted during implementation → `docs/citations.md` (you maintain this)
