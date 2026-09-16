# Development Specification — FreeToken-C++ (Working Name)

**Purpose of this document:** a self-contained brief for an AI coding agent (Claude Code) to begin implementation. It defines scope, architecture, environment, testing, benchmarking, and exposed parameters. This is a specification, not a tutorial — cite official docs inline as you implement so the process stays educational for the author, who is upskilling from an embedded-systems background.

**One-line description:** A lightweight, NVIDIA-only, C++/ggml reimplementation of FreeToken's bandwidth-adaptive MoE serving design (arXiv:2608.16157), extended with a hardware-calibrated scheduler, an async SSD prefetch tier, and a minimal OpenAI-compatible server.

**Design philosophy — read before writing code:**
1. **Lightweight over feature-complete.** Prefer the smallest correct implementation of each mechanism. Do not add abstraction layers "for future flexibility" until a second concrete use case actually demands them.
2. **Correctness of the core mechanism over breadth of features.** The q*-style no-stall CPU/GPU split (Module 3) is the entire point of the project. Everything else is secondary and must not compromise this module's clarity.
3. **Every non-trivial design decision gets a comment citing its source** — a paper, a docs page, or a specific reasoning. This is a stated learning requirement, not optional polish.
4. **Scalable structure, not scalable scope.** The repo layout should allow adding a second model architecture, a second GPU backend, or a second cache policy later without a rewrite — but do not build those things now.

---

## 1. Repository Structure

```
freetoken-cpp/
├── CMakeLists.txt
├── README.md
├── docs/
│   ├── architecture.md          # diagrams + design rationale, citations to paper/docs
│   ├── benchmarking.md          # how to reproduce every reported number
│   └── citations.md             # running log of every external source referenced (see Section 7)
├── third_party/
│   └── ggml/                    # vendored or submodule
├── src/
│   ├── core/
│   │   ├── model_loader/        # GGUF parsing, tensor graph construction
│   │   ├── memory/
│   │   │   ├── host_pool.{h,cpp}      # Host-Resident Pool (DRAM, source of truth)
│   │   │   ├── gpu_cache.{h,cpp}      # GPU-Expert Cache (VRAM)
│   │   │   └── ssd_prefetch.{h,cpp}   # async prefetch tier (conditional module)
│   │   ├── scheduler/
│   │   │   ├── qstar_scheduler.{h,cpp}   # core compute-splitting logic
│   │   │   └── calibration.{h,cpp}        # startup bandwidth measurement
│   │   ├── concurrency/
│   │   │   ├── spsc_queue.{h,cpp}         # lock-free queue (Module 3.8)
│   │   │   └── thread_affinity.{h,cpp}    # pinning/NUMA helpers
│   │   └── engine.{h,cpp}       # public library API surface
│   ├── server/
│   │   ├── http_server.{h,cpp}
│   │   ├── openai_api.{h,cpp}   # /v1/completions, /v1/chat/completions
│   │   └── streaming.{h,cpp}    # SSE
│   ├── harness/                 # agent harness (Phase 4, separate binary/module)
│   └── cli/
│       └── main.cpp             # CLI entrypoint exposing engine as standalone binary
├── benchmark/
│   ├── bench_main.cpp           # benchmarking application (Section 5)
│   ├── scenarios/                # scenario configs (see Section 5.3)
│   └── reports/                  # generated output (gitignored, README explains format)
├── tests/
│   ├── unit/                     # per-module unit tests
│   ├── integration/               # end-to-end forward-pass correctness
│   └── fixtures/                  # small test models/tensors, not full checkpoints
└── scripts/
    ├── setup_env.sh
    ├── download_test_model.sh
    └── run_benchmarks.sh
```

**Why this shape:** each `core/` submodule maps 1:1 to a roadmap phase, so a module can be developed and tested in isolation before wiring into `engine.cpp`. The `benchmark/` and `tests/` separation matters — benchmarks measure performance and produce reports; tests assert correctness and gate CI. Conflating them (a common mistake) makes both harder to trust.

---

## 2. Environment & Toolkit

| Component | Choice | Why |
|---|---|---|
| OS | Ubuntu (native or WSL2) | CUDA + `io_uring` support; see prior WSL2 notes in `docs/architecture.md` |
| Build system | CMake ≥ 3.24 | Standard for C++/CUDA mixed projects; ggml itself uses CMake |
| Compiler | GCC ≥ 11 or Clang ≥ 15 | C++17 minimum (ggml requirement); C++20 acceptable if not blocking |
| GPU toolkit | CUDA Toolkit (match installed driver) | NVIDIA-only scope for v1 |
| Core dependency | ggml (vendored via git submodule, pinned commit) | Do not vendor full llama.cpp — only what's needed from ggml core + backend |
| HTTP | cpp-httplib (header-only) or raw sockets | Header-only keeps the dependency footprint minimal; raw sockets only if the learning goal specifically includes socket programming |
| Async I/O | liburing (for `io_uring`) with `pread` fallback | Detect kernel support at configure time; do not hard-require `io_uring` |
| Testing | Catch2 or GoogleTest (pick one, do not mix) | Both are standard; Catch2 is header-only and lighter |
| Profiling | `perf`, `strace`, NVIDIA Nsight Systems (`nsys`) | External tools, not linked dependencies |
| Package management | None beyond CMake `FetchContent`/submodules | Avoid Conan/vcpkg unless dependency count grows past ~5 — keep it lightweight |

**Pinning discipline:** pin ggml to a specific commit SHA, not a branch. Record the SHA and the date it was pinned in `docs/citations.md`. Re-pinning is a deliberate, logged decision, not a silent `git pull`.

---

## 3. Core Modules — Interfaces & Responsibilities

### 3.1 Model Loader
- Parses GGUF format (reuse ggml's reader, do not reimplement).
- Builds `ggml_cgraph` for the target MoE architecture.
- **Explicitly confirm and document whether ggml's loader uses `mmap` or `read()`** for the GGUF file, and why that matters here: `mmap` enables lazy page-in (weights not yet touched aren't actually read from disk), which interacts directly with the Host-Resident Pool's memory footprint and the OS page cache — this is a genuine Linux-internals decision point, not incidental plumbing. If ggml's default behavior doesn't match what you want (e.g. you want to force-populate pages during calibration rather than lazily), document the override.
- **Exposed parameters:** model path, context length, batch size, quantization format (read from GGUF metadata, not user-specified — the file dictates this).
- **Cite:** GGUF spec (ggml-org/ggml `docs/gguf.md`), ggml tensor/graph API docs, `man 2 mmap`.

### 3.2 Host-Resident Pool (DRAM)
- Holds the complete expert weight set as the source of truth.
- Allocation strategy: pinned/page-locked host memory (`cudaHostAlloc` or `mlock`) so it's directly usable for both CPU compute and async H2D copy without an extra staging copy.
- Use `madvise(MADV_WILLNEED)` when the calibration pass or scheduler can predict near-term access to a region, and `MADV_DONTNEED`/similar when evicting — this isn't limited to the SSD tier; page cache hinting is relevant here too.
- **Exposed parameters:** max host memory budget (bytes or % of system RAM), pinning on/off (pinning improves transfer speed but reduces memory available to the OS — expose this tradeoff, don't hide it), `madvise` hinting on/off.
- **Cite:** CUDA C++ Programming Guide § pinned memory; `man mlock`; `man 2 madvise`.

### 3.3 GPU-Expert Cache (VRAM)
- Dynamic cache of currently-hot experts.
- Eviction policy: start with LRU (simplest correct baseline), add frequency-weighting only after LRU is benchmarked and understood.
- **Exposed parameters:** VRAM budget for the cache (bytes or % of free VRAM), eviction policy selector (`lru` | `frequency`), cache line/slot size.
- **Cite:** the FreeToken paper's cache design section; note any deliberate deviation.

### 3.4 q\*-Style Scheduler (the core contribution)
- On a cache miss: instead of stalling for a synchronous H2D copy, split the token's compute — dispatch to CPU (against the Host-Resident Pool directly) while the GPU continues other work.
- Split decision uses measured bandwidth (from Module 3.6) against a configurable threshold or heuristic — start with a simple threshold, document it as a simplification of FreeToken's closed-form optimal split.
- **Exposed parameters:** split heuristic mode (`threshold` | `closed_form` — closed_form as a stretch goal), CPU thread count dedicated to fallback compute.
- **This module needs the most thorough code comments and citations of anything in the repo** — every deviation from the paper's actual formulation must be explicitly noted, not silently simplified.
- **Cite:** FreeToken paper §"Architecture of Edge-Native MoE Serving" and the q\* policy section specifically.

### 3.5 Async SSD Prefetch Tier (conditional — only if target model exceeds host RAM)
- Never computed against directly — strictly prefetches into the Host-Resident Pool ahead of need.
- Uses MoE's sequential layer execution order as a lookahead signal.
- **Exposed parameters:** prefetch distance (layers ahead), I/O backend (`io_uring` | `pread`, auto-detected with manual override), SSD read queue depth.
- **Cite:** `man 7 io_uring`, liburing examples, Kerrisk's *Linux Programming Interface* (mmap/pread chapters).

### 3.6 Hardware Calibration Pass
- Runs at engine startup: measures actual achievable PCIe H2D/D2H bandwidth and DRAM read/write throughput on the current machine.
- Feeds measured values into Module 3.4's split thresholds.
- **Exposed parameters:** calibration sample size/duration (tradeoff: longer calibration = more accurate but slower startup), option to skip calibration and supply manual bandwidth values (useful for reproducible benchmarking across runs).
- **Cite:** this is your own contribution — document the methodology clearly in `docs/architecture.md` with a comparison to how OSPI PHY tuning solves an analogous problem in the embedded context, since that's the actual justification for the technique.

### 3.7 Engine Public API (library surface)
- The single header/interface other code (server, harness, benchmark app) depends on.
- Should NOT leak ggml internals or CUDA types into the public interface — wrap them.
- **Cite:** llama.cpp's own `llama.h` as a reference for what a clean C++ inference library surface looks like.

### 3.8 Low-Latency Concurrency Primitives (explicit module, not folded into the server)
This is its own component precisely so it gets built and tested deliberately, not assumed via "just use a thread pool."
- **Lock-free (or single-producer/single-consumer at minimum) queue** between the network-facing thread and the inference thread(s) — do not use a mutex-guarded `std::queue` as the final implementation; that's the naive baseline to benchmark against, not the deliverable.
- **Thread affinity:** pin the scheduler's CPU-fallback compute threads (Module 3.4) with `pthread_setaffinity_np`, ideally to cores topologically close to the GPU's PCIe lane if NUMA topology is detected (`numactl --hardware` to inspect).
- **Memory ordering:** document which atomics use `relaxed` vs `acquire/release` vs `seq_cst` and why — this is a common interview probe area, so getting it right and being able to explain it matters more than just having it compile.
- **Exposed parameters:** queue capacity/backpressure policy, thread pinning on/off, number of dedicated inference threads vs I/O threads.
- **Cite:** Martin Thompson's "Mechanical Sympathy" blog; *C++ Concurrency in Action* (Anthony Williams) for the memory-ordering material; moodycamel's concurrentqueue repo as a reference implementation (study it, decide whether to use it directly or hand-roll a simpler SPSC queue for the learning value — hand-rolling is worth more here given the project's learning goals).

---

## 4. Server Layer

- OpenAI-compatible `/v1/completions` and `/v1/chat/completions`.
- SSE streaming for token-by-token output.
- Uses Module 3.8's lock-free queue + pinned thread pool — do not implement a separate, simpler concurrency mechanism here; the server is the primary consumer that justifies 3.8's existence.
- Continuous batching if time allows (documented as stretch goal otherwise).
- **Exposed parameters:** port, max concurrent requests, request timeout, batch window (ms).
- **Cite:** OpenAI API reference (for schema compatibility), llama.cpp's `llama-server` source as a structural reference (not to copy verbatim).

---

## 5. Benchmarking Application

This is a **separate binary** (`benchmark/bench_main.cpp`), not folded into the main engine — it must be runnable against different engine configurations without modifying engine code.

### 5.1 What It Measures
- **Throughput:** tokens/sec (prefill and decode measured separately — they have different bottlenecks).
- **Latency:** time-to-first-token (TTFT), inter-token latency, p50/p95/p99 under both single-request and concurrent load.
- **Cache behavior:** GPU-Expert Cache hit rate, per-tier hit rate if SSD tier is enabled.
- **Scheduler behavior:** CPU/GPU compute-split ratio over time (this is the number that proves the q\* mechanism is doing something, not just present).
- **Resource usage:** peak VRAM, peak host RAM, calibration overhead (startup time cost).

### 5.2 Comparison Modes (must support running all of these from one tool, same model/hardware)
1. **Naive baseline:** synchronous stall-and-copy on cache miss (no q\* logic) — this is the control.
2. **q\* without calibration:** fixed/assumed bandwidth constants.
3. **q\* with calibration:** your full Module 3.6 contribution.
4. **With vs without SSD prefetch tier**, only on a model sized to exceed host RAM.

A single command should produce a comparison table/report across these modes for a given model+hardware pairing — this is what generates your actual resume/portfolio numbers, so it needs to be trustworthy and repeatable, not a one-off script.

### 5.3 Scenario Configs
Store benchmark scenarios as config files (`benchmark/scenarios/*.yaml` or `.json`), not hardcoded in source — e.g. `scenarios/small_model_8gb_gpu.yaml`. Each scenario specifies: model path, GPU/host memory budgets, concurrency level, request pattern (single vs sustained load), and which comparison modes to run. This makes results reproducible and is itself a good engineering practice to point to in interviews.

### 5.4 Output Format
- Machine-readable (JSON/CSV) as the source of truth, human-readable table as a summary view.
- Every benchmark run should log: git commit SHA, hardware (`nvidia-smi` output, CPU model, RAM size), and the scenario config used — reproducibility discipline.

---

## 6. Testing Strategy

| Level | What it covers | Tooling |
|---|---|---|
| **Unit** | Individual module correctness (e.g. cache eviction logic, calibration math, GGUF parsing on a tiny fixture file) | Catch2/GoogleTest, run on every commit |
| **Integration** | Full forward pass produces numerically correct output vs a known-good reference (e.g. compare against llama.cpp's own output on the same small model, within a floating-point tolerance) | Custom harness, run on every PR |
| **Regression/benchmark gating** | Not full benchmarking, but a lightweight "did this change tank throughput by >X%" smoke check | Run the naive baseline scenario only, on CI hardware if GPU CI is available, otherwise document as manual pre-merge step |
| **Concurrency/load** | Module 3.8's lock-free queue under contention (no lost/duplicated items, correct ordering where required); server layer under concurrent requests — no crashes, no deadlocks, correct interleaving of streamed responses | ThreadSanitizer (`-fsanitize=thread`) for the queue specifically, plus a load-testing script (`scripts/run_benchmarks.sh` can include a light load test, or use a tool like `wrk`/`hey`) for the server |

**Important scoping note:** full GPU-in-CI is often impractical/expensive for a solo project. It's acceptable and worth documenting explicitly that GPU-dependent tests are run manually before releases/milestones, while CPU-only unit tests run in standard CI (GitHub Actions).

---

## 7. Documentation & Citation Requirements (for the author's learning)

This is a stated project requirement, not optional:

- **`docs/citations.md`**: a running, dated log of every external source consulted during implementation — paper sections, official docs pages, RFC/discussion threads, Stack Overflow answers for tricky API usage. Format: `[date] [module] [source URL or citation] — [what was learned/used]`.
- **Inline comments citing sources** for any non-obvious design decision — especially in `qstar_scheduler.cpp` and `calibration.cpp`, where deviations from the FreeToken paper must be explicitly flagged as simplifications.
- **`docs/architecture.md`**: written for a future reader (including future-you) who hasn't read the FreeToken paper — explain the design, cite the paper, explain deviations, include a diagram.
- When Claude Code implements a module, it should be instructed to output a short "what I referenced and why" note per module, not just code — this is what makes the process pedagogically useful rather than just producing a black-box result.

---

## 8. Development Workflow (Suggested Order for the Coding Agent)

1. Scaffold repo structure (Section 1) and CMake build with ggml as a submodule — confirm it builds and links, no functional code yet.
2. Implement Model Loader (3.1) — load a tiny GGUF fixture, confirm a forward pass runs via ggml directly (no custom scheduler yet). **Profiling checkpoint:** run under `strace -c` to see the actual syscall pattern for loading (mmap calls, page faults), and note the observed behavior in `docs/citations.md` — this is the concrete "watch your own project" learning step, not optional.
3. Implement Host-Resident Pool (3.2) and GPU-Expert Cache (3.3) with a naive synchronous stall-and-copy fallback — this is the control/baseline the rest of the project is measured against.
4. Implement Low-Latency Concurrency Primitives (3.8) in isolation first — unit-test the lock-free/SPSC queue under concurrent load *before* wiring it into the scheduler or server, since a concurrency bug found here is far easier to isolate than one found after integration.
5. Implement the q\* Scheduler (3.4) with a fixed-threshold split, using 3.8's queue for CPU/GPU work dispatch — benchmark against the Phase 3 baseline immediately, don't wait until "everything's ready." **Profiling checkpoint:** use `perf stat`/`perf record` to confirm the CPU-fallback path is actually running on its pinned cores and check for unexpected lock contention.
6. Implement the Calibration Pass (3.6), wire it into the scheduler's threshold, re-benchmark.
7. Build the Benchmarking Application (Section 5) alongside steps 3–6, not after — you need it to validate each step's claims as you go, not just at the end.
8. Only after 3–7 are solid: SSD Prefetch Tier (3.5), Server Layer (Section 4), Library API polish (3.7), Agent Harness. **Profiling checkpoint for the server:** load-test with concurrent requests under `strace -f -c` to inspect syscall overhead, and `perf` to check for contention on the 3.8 queue under real concurrent load, not just the earlier unit-test load.
9. Documentation pass (Section 7) throughout, not bolted on at the end — cite as you go, don't try to reconstruct sources afterward.

---

## 9. Explicit Non-Goals (v1)

State these clearly so the coding agent doesn't scope-creep:
- No AMD/ROCm support.
- No multi-GPU/multi-node support.
- No training or fine-tuning — inference only.
- No full closed-form optimal q* split in v1 — threshold heuristic first, closed-form as documented future work.
- No custom quantization formats — consume whatever GGUF quantization is already present in the model file.
- No web UI — CLI and API only.

---

## 10. Deferred for Post-v1 (Not a Non-Goal — Just Sequenced Later)

Distinct from Section 9: these are things we *do* want, just not now. Listed here so they survive to whenever we get there instead of getting rediscovered from scratch.

- **Distribution via Docker image.** The engine as designed (Section 2) is Ubuntu-only (native or WSL2) because the Host-Resident Pool and SSD prefetch tier use Linux-only syscalls (`mlock`, `madvise`, `io_uring`) with no Windows codepath planned. A Docker image would let a Windows end user run the finished engine via `docker run` instead of manually setting up WSL2 + the build toolchain — it doesn't remove the Linux dependency, it packages it. When this gets built, the image/run instructions must account for:
  - GPU passthrough: `--gpus all` (NVIDIA Container Toolkit).
  - `mlock` for the Host-Resident Pool: containers default to a tiny `memlock` ulimit (often 64KB) — needs `--ulimit memlock=-1` or `--cap-add=IPC_LOCK` at run time, or the pool's pinning silently fails.
  - `io_uring` for the SSD tier: blocked by Docker's default seccomp profile regardless of kernel support — needs a custom seccomp profile or `--security-opt seccomp=unconfined`. The kernel-support probe (Section 2, "Async I/O" row) should treat a seccomp `EPERM` the same as "unsupported" and fall back to `pread`, not crash.
