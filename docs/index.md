# FreeToken-C++

A lightweight, NVIDIA-only, C++/ggml reimplementation of [FreeToken](https://arxiv.org/abs/2608.16157)'s
bandwidth-adaptive MoE serving design, extended with a hardware-calibrated
scheduler and (planned) an async SSD prefetch tier.

**This site is generated documentation.** It combines:
- **API reference** — extracted from the `///` comments in `src/`'s headers (browse via *Classes* / *Files* above).
- **Narrative docs** — the project's own markdown files, listed below.
- **Tests** — `tests/` is included in the scan too, so test files show up in the *Files* tree alongside the code they test.

Regenerate this site any time with `doxygen Doxyfile` from the repo root (output goes to `docs_site/`, gitignored — not committed, since it's fully reproducible from source).

## ⚠️ Read this first: what's real vs. placeholder

Before trusting any module as "done," check **[PROGRESS.md](../PROGRESS.md)'s placeholder ledger** at the very top — it lists every stand-in in the codebase (e.g. the CPU/GPU "compute" paths are still a trivial placeholder op, not real MoE math) and what would need to change to make it real. This is the single most important thing to read before relying on any function in the API reference below.

## Where things live

| Document | What's in it |
|---|---|
| [roadmap.md](../roadmap.md) | Why this project exists, the phased plan, what's ported vs. original vs. genuinely novel |
| [dev_spec.md](../dev_spec.md) | Per-module engineering spec — interfaces, exposed parameters, required citations, non-goals |
| [PROGRESS.md](../PROGRESS.md) | Current actual state, the placeholder ledger, what's next |
| [CLAUDE.md](../CLAUDE.md) | Working agreement for how this project gets built (session continuity, teaching-mode chunking) |
| [docs/building.md](building.md) | CMake primer and full build command reference |
| [docs/citations.md](citations.md) | Every external source consulted, and every real mistake made and fixed along the way |

## Module map

Each module below corresponds to a section in `dev_spec.md` — click through to its class/file docs for the API itself.

- **Model Loader** (`src/core/model_loader/`) — GGUF parsing, real MoE expert extraction
- **Host-Resident Pool** (`src/core/memory/host_pool.h`) — the complete expert set in pinned host RAM
- **GPU-Expert Cache** (`src/core/memory/gpu_cache.h`) — the LRU VRAM cache, stall-and-copy baseline
- **SPSC Queue** (`src/core/concurrency/spsc_queue.h`) — the lock-free CPU/GPU dispatch primitive
- **Hardware Calibration** (`src/core/scheduler/calibration.h`) — measures real `B_P`/`B_H` on the deployed machine
- **q\* Scheduler** (`src/core/scheduler/qstar_scheduler.h`) — the actual core mechanism this whole project exists to build

## Running the tests yourself

`scripts/run_benchmarks.sh` builds and runs every test/benchmark in one go, printing a summary with real numbers (bandwidths, timings, speedups) and saving the full log to `benchmark/reports/`.
