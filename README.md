# FreeToken-C++

A lightweight, NVIDIA-only, C++/ggml reimplementation of [FreeToken](https://arxiv.org/abs/2608.16157)'s bandwidth-adaptive MoE serving design — a q\*-based scheduler that splits each decode step's cache-missing experts between a GPU-fill path and direct CPU execution, sized by real measured hardware bandwidth, instead of stalling on synchronous PCIe transfers.

This is a from-scratch C++/ggml port and extension of FreeToken's design (paper cited above), not an original serving architecture — see [`docs/roadmap.md`](docs/roadmap.md) §2a for exactly what's ported vs. genuinely extended.

**Before relying on any part of this as "done," read [`docs/PROGRESS.md`](docs/PROGRESS.md)'s placeholder ledger at the top** — it tracks every stand-in in the codebase (e.g. the actual MoE compute kernel doesn't exist yet; the scheduling mechanism around it does) so nothing gets mistaken for more finished than it is.

## Getting Started

### Prerequisites

- **Linux — WSL2 Ubuntu on Windows, or native Linux.** Native Windows won't work: the design depends on `mlock`/`madvise`/`io_uring`, which don't exist there. See [`docs/roadmap.md`](docs/roadmap.md) §6 for WSL2 setup notes.
- **An NVIDIA GPU** (CUDA compute-capable) — this project is NVIDIA-only by design (no AMD/ROCm).
- Toolchain:
  ```bash
  sudo apt-get install -y build-essential cmake python3-venv doxygen graphviz
  ```
  (`python3-venv` is needed for the `hf` CLI used to fetch a test model; `doxygen`/`graphviz` for the docs site.)
- **CUDA Toolkit** matching your driver (`nvcc --version` to check; `sudo apt-get install -y nvidia-cuda-toolkit` if missing).
- **Hugging Face CLI**, for downloading the real test MoE model:
  ```bash
  curl -LsSf https://hf.co/cli/install.sh | bash -s
  ```

### Clone and build

```bash
git clone --recurse-submodules <this-repo-url>
cd freetoken-cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

(If you forgot `--recurse-submodules`, run `git submodule update --init` afterward — `third_party/ggml` is a submodule, not vendored source.)

### Download the test model

One real (tiny, random-weight) MoE GGUF model is used by the integration tests, not committed to git (binary, regenerable):
```bash
bash scripts/download_test_model.sh
```

### Run the tests

```bash
ctest --test-dir build              # everything
ctest --test-dir build -V           # everything, with each test's printed numbers (bandwidths, timings)
ctest --test-dir build -R QStarScheduler   # just one suite
```

Or build and run everything with a consolidated numbers summary saved to `benchmark/reports/`:
```bash
bash scripts/run_benchmarks.sh
```

### Generate the docs

Combines the API reference (from the code's own `///` comments), every markdown doc, and the test tree into one browsable HTML site:
```bash
bash scripts/build_docs.sh
```
Opens from `index.html` at the repo root (a redirect into `docs_site/html/index.html`, the real site).

## Project layout

```
CMakeLists.txt          # root build: CUDA, GoogleTest (FetchContent), pulls in everything below
src/
├── cli/                 # freetoken-cli entrypoint
└── core/
    ├── model_loader/     # GGUF parsing, real MoE expert extraction
    ├── memory/           # Host-Resident Pool (pinned RAM) + GPU-Expert Cache (VRAM, LRU)
    ├── concurrency/       # lock-free SPSC queue (CPU/GPU dispatch primitive)
    └── scheduler/         # hardware calibration pass + the q* scheduler itself
tests/
├── unit/<module>/        # pure-logic tests, mirrors src/core/<module>/
└── integration/<module>/ # real GPU/pool/cache/model tests, same mirroring
docs/                     # roadmap, engineering spec, progress log, citations, this site's landing page
scripts/                  # download_test_model.sh, run_benchmarks.sh, build_docs.sh
```

## Where to read next

| Doc | What's in it |
|---|---|
| [`docs/roadmap.md`](docs/roadmap.md) | Why this project exists, the phased plan, what's ported vs. original |
| [`docs/dev_spec.md`](docs/dev_spec.md) | Per-module engineering spec — interfaces, exposed parameters, citations, non-goals |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | Current actual state, the placeholder ledger, what's next |
| [`docs/building.md`](docs/building.md) | Full CMake/testing/docs command reference |
| [`docs/citations.md`](docs/citations.md) | Every external source consulted, and every real mistake made and fixed along the way |
| [`CLAUDE.md`](CLAUDE.md) | Working agreement for how this project gets built with Claude Code |
