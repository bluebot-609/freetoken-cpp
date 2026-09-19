# FreeToken-C++

A lightweight, NVIDIA-only, C++/ggml reimplementation of [FreeToken](https://arxiv.org/abs/2608.16157)'s
bandwidth-adaptive MoE serving design, extended with a hardware-calibrated
scheduler and (planned) an async SSD prefetch tier.

## How to navigate this site

This is generated documentation (Doxygen) — regenerate it any time with `doxygen Doxyfile` from the repo root (output: `docs_site/html/index.html`, gitignored, fully reproducible from source). The sidebar on the left has four things worth knowing about:

- **Related Pages** — every markdown doc in this project (this page, `README.md`, `CLAUDE.md`, everything under `docs/`), rendered as its own page. Start here for *why*.
- **Classes** — every `struct`/`class` (e.g. `HostResidentPool`, `QStarScheduler`), with member lists and full doc-comment text.
- **Files** — the actual source tree, `src/` and `tests/` together, so a test file sits alongside the code it exercises. Click a file to see its declarations; click **source** on any page to see the real syntax-highlighted code, cross-referenced (click a function call to jump to its definition).
- **Search** (top right) — fastest way to jump straight to a function/class/file by name.

If you're new here, read in this order: this page → [roadmap.md](roadmap.md) (why) → [dev_spec.md](dev_spec.md) (how each module is specced) → [PROGRESS.md](PROGRESS.md) (what's actually built right now) → then browse **Classes**/**Files** for the API itself.

## ⚠️ Read this first: what's real vs. placeholder

Before trusting any module as "done," check **[PROGRESS.md](PROGRESS.md)'s placeholder ledger** at the very top — it lists every stand-in in the codebase (e.g. the CPU/GPU "compute" paths are still a trivial placeholder op, not real MoE math) and what would need to change to make it real. This is the single most important thing to read before relying on any function in the API reference.

## Where things live

| Document | What's in it |
|---|---|
| [README.md](../README.md) | Getting started — clone, build, test, generate docs |
| [roadmap.md](roadmap.md) | Why this project exists, the phased plan, what's ported vs. original vs. genuinely novel |
| [dev_spec.md](dev_spec.md) | Per-module engineering spec — interfaces, exposed parameters, required citations, non-goals |
| [PROGRESS.md](PROGRESS.md) | Current actual state, the placeholder ledger, what's next |
| [CLAUDE.md](../CLAUDE.md) | Working agreement for how this project gets built (session continuity, teaching-mode chunking) |
| [building.md](building.md) | CMake primer and full build/test/docs command reference |
| [citations.md](citations.md) | Every external source consulted, and every real mistake made and fixed along the way |

## Module map

Each module below corresponds to a section in `dev_spec.md` — click through to its class/file docs for the API itself.

- **Model Loader** (`src/core/model_loader/`) — GGUF parsing, real MoE expert extraction
- **Host-Resident Pool** (`src/core/memory/host_pool.h`) — the complete expert set in pinned host RAM
- **GPU-Expert Cache** (`src/core/memory/gpu_cache.h`) — the LRU VRAM cache, stall-and-copy baseline
- **SPSC Queue** (`src/core/concurrency/spsc_queue.h`) — the lock-free CPU/GPU dispatch primitive
- **Hardware Calibration** (`src/core/scheduler/calibration.h`) — measures real `B_P`/`B_H` on the deployed machine
- **q\* Scheduler** (`src/core/scheduler/qstar_scheduler.h`) — the actual core mechanism this whole project exists to build

## Tests

`tests/unit/<module>/` and `tests/integration/<module>/` mirror `src/core/<module>/` — a test's folder tells you what it covers. Every test is a real [GoogleTest](https://github.com/google/googletest) case with its own ID (`TestSuite.TestName`, e.g. `QStarScheduler.FasterThanNaiveStallAndCopyBaseline`), registered with CTest.

```bash
ctest --test-dir build              # run everything
ctest --test-dir build -R QStarScheduler   # run one suite
bash scripts/run_benchmarks.sh      # build + run everything + a numbers summary, saved to benchmark/reports/
```
