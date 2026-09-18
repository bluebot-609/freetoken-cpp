# MoE Inference Engine — Skill Map, Roadmap & Resources

A standalone C++ inference engine (using ggml as a library, NVIDIA-only for now) that ports and extends **FreeToken**'s edge-native MoE serving design, plus a low-latency OpenAI-compatible server and an agent harness on top.

---

## 0. Source of Truth: FreeToken

**Repo:** github.com/FlashML-org/FreeToken
**Paper:** arXiv:2608.16157 — "FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution" (Yang, Fan, Pan, Xi, Wang, Sun, Keutzer, Han, Zaharia, Xu, Stoica — UT Austin / UC Berkeley)

**Core idea (read this before writing any code):** FreeToken does NOT rely on simple hot-expert caching with synchronous fallback. Its central mechanism is the **q\* policy** — at each decode step, of the set of experts that missed the GPU cache (size `m`), it partitions them into a GPU-fill subset `F` (size `q`, transferred over PCIe into the cache) and a CPU-direct-execute subset `C` (size `m-q`, computed in place on the CPU), run **concurrently**, then merges their partial outputs exactly. This is a **per-step scheduling decision over a set of missing experts**, not "splitting one token's math in half between two processors" — a distinction that matters for implementing Module 3.4 correctly. The split size is a genuinely simple closed form, not something requiring an interim heuristic: `q* ≈ m · (B_P / B_H)`, where `B_P` is measured PCIe transfer bandwidth and `B_H` is measured CPU-side expert-processing bandwidth (paper §3.2, Eq. 1–4). **Both bandwidths are empirically profiled on the actual deployed hardware at startup** — this detail matters, see the correction below.

**Memory architecture as designed (2-tier, not 3):**
- **Host-Resident Pool (DRAM):** the complete set of expert weights — the "source of truth."
- **GPU-Expert Cache (VRAM):** a dynamic, shared cache of currently-hot experts.
- FreeToken deliberately stops at DRAM as the floor — it assumes host RAM is large enough to hold the full model, which covers most workstations/gaming PCs.

**CORRECTION (2026-09-19, after actually reading the paper — `docs/references/freetoken_arxiv_2608.16157.pdf`):** the hardware calibration pass previously listed below as "the headline novel contribution... not described in the paper" is **already in the paper**. §3.2 states the bandwidth parameters `B_H` and `B_P` "are empirically profiled on the target hardware at deployment," and Table 1's caption confirms all bandwidths were measured on deployed hardware, not taken from spec sheets. Porting this correctly is real, valuable work — it just isn't *original* work, and claiming otherwise in an interview to someone who's read the paper would be an easily-caught mistake. See the corrected novelty table in §2a.

**Our scoped extensions beyond a straight port** (see Section 2a for full rationale):
1. **Async SSD prefetch tier** — for models exceeding host DRAM, add a third tier that prefetches experts from disk into DRAM ahead of need (layer-order lookahead), never computed against directly. Only build/benchmark this if the target model+hardware pairing genuinely exceeds host RAM — otherwise it's unfalsifiable in your own benchmarks. (Confirmed still genuinely absent from the paper: FreeToken's expert pool is loaded from disk once at startup and never used as an execution/prefetch tier.)
2. **Library-first API** — expose the engine as a linkable C++ library (like llama.cpp itself), not just a server process.
3. **Static schedule mode (optional)** — precompute a per-layer CPU/GPU split once via profiling, instead of deciding every token, for fixed-deployment scenarios where FreeToken's real-time adaptivity isn't needed.

**Citation discipline:** every writeup, README, and interview answer should describe this as "a C++/ggml, NVIDIA-only port and extension of FreeToken's q\* co-execution design," with the paper cited. Claiming the caching/splitting idea — or the calibration pass — as original would be inaccurate and would read poorly to anyone in this space.

---

## 1. Skill Map — Project Layer → Required Skills

| Layer | What you build | Skills exercised | Your existing edge |
|---|---|---|---|
| **Model loading** | GGUF parser, tensor graph via ggml | C++, binary file formats, mmap | Familiar with binary/register-level thinking from embedded |
| **Bandwidth calibration** | Startup pass measuring real PCIe/DRAM throughput | Benchmarking methodology, low-level timing | Direct parallel to OSPI PHY tuning bring-up |
| **q\*-style scheduler** | Per decode-step, partition the set of missing experts between GPU-fill and CPU-direct-execute using measured bandwidth (`q* ≈ m·B_P/B_H`), run concurrently, merge exactly | C++, real-time scheduling logic, no-stall design | EDMA-style "move data to where compute happens" thinking |
| **GPU-Expert Cache (VRAM)** | Dynamic, shared hot-expert cache | CUDA basics, memory allocation, async copy | New — but memory-tier concept is familiar |
| **Host-Resident Pool (DRAM)** | Full expert set as source of truth, CPU-computable fallback | Linux virtual memory, `mlock`/`cudaHostAlloc`, DMA | Directly maps to your PSRAM/OCRAM offload work |
| **SSD prefetch tier (extension)** | Async layer-lookahead prefetch into DRAM, never computed against directly | `pread`/`io_uring`, async I/O, page cache behavior | Directly maps to your flash/QSPI experience |
| **Cache eviction policy** | LRU/frequency-weighted GPU cache management | Data structures, algorithm design, benchmarking rigor | Your DSA background is a direct asset here |
| **Low-latency server** | OpenAI-compatible HTTP/SSE API | Concurrency, thread pools, lock-free queues, network programming | Low-latency mindset transfers from OSPI/EDMA optimization |
| **Agent harness** | ReAct-style tool-calling loop | Python, API design, orchestration | New territory — fastest to pick up |
| **Profiling/measurement** | Latency, cache-hit-rate, CPU/GPU-split ratio instrumentation | `perf`, `strace`, `/proc`, `nsys`/`nvprof` | New tools, familiar profiling *mindset* |

---

## 2. Roadmap (Phased, No Timelines — Sequenced by Dependency)

### Phase 0 — Foundations
Close gaps before they block you mid-build. Can run partly in parallel with early Phase 1 reading.
- Modern C++ (smart pointers, move semantics, templates, RAII) — you already know C deeply, this is fast.
- CUDA basics: memory hierarchy, kernel launch model, `cudaMemcpyAsync`, pinned memory.
- Read ggml's source structure (not full internals) — understand `ggml_tensor`, `ggml_cgraph`, backend scheduler (`ggml-backend.cpp`).
- Read the FreeToken paper closely (not just the README) — understand the q\* formulation before writing any scheduler code.

**Exit criterion:** a "hello world" ggml program that loads a small GGUF model and runs a forward pass with no custom caching — just to prove the plumbing works.

### Phase 1 — Single-model correctness, GPU-Expert Cache + DRAM pool (FreeToken's core, ported)
The core of the project. Do not move on until this is solid and benchmarked.
- Model loader for a small/quantized MoE model (start small to prove the mechanism before scaling up).
- Host-Resident Pool: full expert set loaded into DRAM as source of truth.
- GPU-Expert Cache: dynamic VRAM cache with an LRU or frequency-weighted eviction policy.
- **q\*-style scheduler (the actual core contribution to port correctly):** on a cache miss, split that token's compute between CPU and GPU rather than stalling on a synchronous copy. Start with a simplified heuristic split (e.g. a measured-bandwidth threshold) before attempting a full closed-form optimal split — a correct simple version beats an incorrect sophisticated one.
- Baseline comparison: naive "always stall and copy from host" vs your no-stall split.
- Instrumentation: cache hit rate, CPU/GPU compute-split ratio, tok/s, p50/p99 latency.

**Exit criterion:** working engine with a benchmark report showing the no-stall split beating the naive synchronous baseline. This alone is a complete, demoable project if you need to stop here.

### Phase 1b — Hardware calibration pass (headline novel contribution)
- Startup routine that measures actual PCIe transfer bandwidth and DRAM throughput on the specific machine, rather than assuming fixed/estimated values.
- Feed these measured numbers into the q\*-style split thresholds.
- Frame explicitly as: an OSPI-PHY-tuning-style hardware bring-up technique applied to ML systems scheduling.

**Exit criterion:** benchmark showing the calibrated version outperforming (or at least matching with lower variance) a version using fixed/assumed bandwidth constants.

### Phase 2 — Async SSD prefetch tier (extension, conditional)
Only pursue this if your benchmark model+hardware pairing genuinely exceeds host DRAM — otherwise this phase is unfalsifiable in your own numbers.
- SSD tier via `pread` or `io_uring`, used strictly for **async prefetch into DRAM**, never computed against directly (SSD is too slow for FreeToken's no-stall CPU fallback trick to work against it directly).
- Layer-order lookahead: issue prefetch reads 1–2 layers ahead of need, based on MoE's sequential layer execution.
- `madvise` hints, page cache observation.
- Re-benchmark 3-tier vs 2-tier, specifically on a model too large for DRAM.

### Phase 3 — Low-latency server
Can start once Phase 1's engine is stable enough to call as a library.
- OpenAI-compatible `/v1/completions`, `/v1/chat/completions`.
- SSE streaming, request queue, thread pool, continuous batching.
- Load test: concurrent requests, measure p50/p99 under load.

### Phase 3b — Library-first API (extension)
- Expose the engine as a linkable C++ library with a clean API, not just an HTTP server process — mirrors how llama.cpp itself is consumed by other native applications.

### Phase 4 — Agent harness
Independent of the engine internals — can be prototyped early against any OpenAI-compatible endpoint, then pointed at your own server once Phase 3 is ready.
- ReAct loop, 2–3 tools, error handling/retries.
- Point it at your server via the API.

### Phase 4b — Static schedule mode (optional extension)
- Profile once, bake a static per-layer CPU/GPU split table, and offer this as an alternative to real-time q\*-style decisions for fixed-deployment scenarios where per-token scheduling overhead isn't worth paying.

### Phase 5 — Polish & narrative
- Write up the project (README with architecture diagram, benchmark numbers, design rationale).
- Cite FreeToken explicitly and correctly (paper + repo) as the design this project ports and extends.
- Prepare the "what I'd do differently / what's next" section — interviewers like this.

### Phase 5b — Distribution via Docker image (deferred, see `dev_spec.md` §10)
- Package the built engine as a Docker image so a Windows end user can `docker run` it instead of setting up WSL2 manually — this packages the Linux dependency, it doesn't remove it (Docker Desktop on Windows still runs on WSL2 underneath).
- Must document/bundle: `--gpus all` for GPU passthrough, `--ulimit memlock=-1` (or `--cap-add=IPC_LOCK`) so the Host-Resident Pool's `mlock` doesn't silently fail against Docker's default tiny memlock ulimit, and a seccomp allowance for `io_uring` (custom profile or `--security-opt seccomp=unconfined`), since Docker's default seccomp profile blocks it regardless of kernel support.

---

## 2a. Novelty Summary — What's Actually Yours

| Contribution | FreeToken has this? | Why it's defensible as "yours" |
|---|---|---|
| C++/ggml, NVIDIA-only implementation | No (Python) | Genuine reimplementation in a different systems stack |
| q\* scheduling policy (fetch-set/compute-set split) | Yes — ported, not invented | Correctly porting a subtle mechanism (§3.2, Eq. 1–4) is real engineering work, but credit the source |
| Hardware calibration pass (measuring `B_P`/`B_H` on deployed hardware) | **Yes — ported, not invented.** CORRECTED 2026-09-19 after reading the paper: §3.2 explicitly says these bandwidths are "empirically profiled on the target hardware at deployment." Previously miscategorized here as an original contribution — it isn't. | Still real, valuable engineering work — porting a load-bearing measurement step correctly and reasoning about it from an embedded hardware-characterization background — but credit the source, same as the q\* port |
| Async SSD prefetch tier | No — paper stops at DRAM | Genuine extension, conditional on model exceeding host RAM |
| Library-first API | Partial (Python package) | Different consumption model, legitimate design choice |
| Static schedule mode | No — paper assumes dynamic agent workloads | A deliberate alternative design point for fixed deployments |

**How to talk about this in interviews:** describe the q\* port (including its calibration step) as "correctly understanding and reimplementing a research system's core mechanism in C++" — that's the actual, defensible skill signal here, not a false originality claim. Lead with the SSD prefetch tier as the genuine extension, and mention library API / static mode as secondary. Do NOT claim the calibration pass as original — it was checked directly against the paper and isn't.

---

## 3. Resources Per Layer

### C++ (modern)
- *Effective Modern C++* — Scott Meyers (the standard reference for move semantics, smart pointers)
- cppreference.com — for API-level lookups while coding

### ggml / GGUF internals
- ggml-org/llama.cpp source itself — read `ggml/src/ggml-backend.cpp` and `ggml/src/ggml.c`
- ggml-org/ggml repo's examples directory — smaller, more digestible than full llama.cpp
- The GitHub discussions you already found (search terms: "MoE expert cache llama.cpp", "moe-expert-cache RFC") — read these threads fully, they contain real design tradeoffs

### CUDA
- *Programming Massively Parallel Processors* (Kirk & Hwu) — the standard textbook
- NVIDIA's official CUDA C++ Programming Guide (docs.nvidia.com)
- NVIDIA CUDA samples repo (github.com/NVIDIA/cuda-samples)

### Linux internals (memory, I/O)
- *Linux Kernel Development* — Robert Love (readable, not overly deep)
- *The Linux Programming Interface* — Michael Kerrisk (essential for `mmap`, `pread`, `io_uring`-adjacent syscalls, memory locking)
- `man 7 io_uring`, and the liburing examples repo (github.com/axboe/liburing)
- Brendan Gregg's blog (brendangregg.com) — for `perf`, flame graphs, and systems profiling methodology

### Low-latency / concurrency
- "Mechanical Sympathy" blog (Martin Thompson) — canonical for cache-aware, low-latency design
- *C++ Concurrency in Action* — Anthony Williams
- moodycamel's lock-free queue (github.com/cameron314/concurrentqueue) — good reference implementation to study (or use directly)

### MoE offloading prior art (read before/while designing your policy)
- **FreeToken paper (arXiv:2608.16157) — primary source of truth, read this first and closely.** Focus on the q\* policy formulation and the two-level expert-memory hierarchy (Host-Resident Pool + GPU-Expert Cache) sections.
- **FlashML-org/FreeToken** repo (Python) — not a direct code reference for your C++ port, but read `python/freetoken` for how they structure the scheduler and cache logic conceptually.
- MoE-Infinity paper (arxiv.org/abs/2401.14361) — activation-aware expert caching, useful secondary reference/comparison point
- The llama.cpp GitHub discussions/RFCs you already surfaced — search "moe-expert-cache", "hot expert cache", "expert-aware SSD streaming" — useful for the SSD prefetch tier specifically, since FreeToken doesn't cover disk

### Server / networking
- A minimal reference: study how llama.cpp's own `llama-server` implements streaming — good scaffold to understand before building your own
- cpp-httplib or writing raw sockets — pick based on how much you want to learn networking internals vs move fast

### Agent harness
- ReAct paper (arxiv.org/abs/2210.03629) — origin of the pattern
- Read one framework's source (LangGraph or a minimal open-source ReAct implementation) for structure ideas, but hand-roll your own for the learning value

---

## 3b. GitHub Reference Repos — Inference Engines to Study

Ordered roughly from "read for architecture ideas" to "read for minimal/readable implementation."

**Production-grade, for architecture patterns (don't try to read cover-to-cover):**
- **ggml-org/llama.cpp** — your base dependency; also the closest reference for GGUF loading, quantization, and the `ggml-backend.cpp` scheduler you'll be intercepting/extending.
- **ggml-org/ggml** — the core tensor library on its own, smaller and easier to read than full llama.cpp; good for understanding `ggml_tensor`/`ggml_cgraph` in isolation.
- **vllm-project/vllm** — Python/CUDA, not C++, so not a direct code reference, but essential reading for *concepts*: PagedAttention and KV-cache-as-virtual-memory is the conceptual sibling of what you're doing with expert caching. Read the design docs/blog posts even if you don't read the Python source closely.
- **huggingface/text-generation-inference (TGI)** — another production server worth skimming for how they structure the serving layer (batching, request handling).

**MoE-specific offloading — direct prior art for your caching layer:**
- The llama.cpp GitHub **discussions/issues** you already found are more valuable here than any single repo — search within ggml-org/llama.cpp discussions for: "moe-expert-cache", "hot expert cache", "expert-aware SSD streaming", "MoE offload disk paging". These contain real design debates (LRU vs frequency-weighted, slot remapping, sentinel-skip flags) that most blog posts won't cover.
- **TorchMoE/MoE-Infinity** — activation-aware expert offloading/caching, closest academic-grade reference for your promotion policy design; also has a citable paper.
- Community forks referenced in those discussions (e.g. forks implementing per-layer hot expert caching in `ggml-backend.cpp`) — useful as "here's one way someone did it," explicitly not meant to be copied, but good for spotting design mistakes to avoid.

**Small/minimal, for reading an entire inference pipeline end-to-end quickly:**
- Search GitHub for small "minimal C++ LLM inference" or "llm.cpp"-style educational repos (several exist with names like `easy_llm.cpp`, `mobilellama`, `uLLM`) — these are far smaller than llama.cpp and let you see tokenizer → prefill → decode → sampling in one sitting. Quality and maintenance vary a lot (check last-commit date and star count before trusting one as a reference), but they're useful as a "what's the minimum viable inference loop" sanity check before you dive into ggml's more complex scheduler.
- **karpathy/llama2.c** — Andrej Karpathy's ~1000-line pure-C inference implementation. Not ggml-based and not MoE, but genuinely the clearest "here is a full forward pass with no framework magic" reference available. Good to read in Phase 0 before touching ggml.

**A note on currency:** this space moves fast — star counts, active-maintenance status, and even which projects are considered "the reference" shift within months. Before committing significant time to studying any single repo, check its last commit date and open issues to confirm it's still actively representative of current practice.

## 4. Essentials Checklist (Tooling & Hardware)

- **Hardware:** a machine with an NVIDIA GPU (even a modest consumer one — 8–12GB VRAM is enough to demonstrate offloading, since the whole point is the model doesn't fit). Cloud GPU rental (RunPod, Lambda, Vast.ai) is a fine substitute if you don't own one.
- **OS:** Linux (Ubuntu is the path of least resistance for CUDA + io_uring support).
- **Toolchain:** CMake, a recent GCC/Clang, CUDA Toolkit matching your driver version.
- **Profiling tools:** `perf`, `strace`, `nsys` (NVIDIA Nsight Systems) for GPU-side timeline profiling.
- **Model:** start with a small/quantized MoE GGUF you can download (search Hugging Face for GGUF MoE models — Qwen3-30B-A3B GGUF quantized versions exist and are commonly referenced in this space).
- **Version control discipline:** clean commit history + a detailed README — this becomes your interview artifact, treat it like a product.

---

## 5. Priority Order If You Need to Stop Early

If you have to start interviewing before finishing everything:
1. Phase 0 + Phase 1 (q\*-style no-stall scheduler, benchmarked against naive stalling) — **non-negotiable, this is the core ported mechanism.**
2. Phase 1b (hardware calibration pass) — **your headline novel contribution, prioritize this over anything in Phase 2–4.**
3. Phase 3 (server) — makes it demoable as a real service, not just a benchmark script.
4. Phase 4 (harness) — cheapest to add, do this if you have any spare capacity before interviews.
5. Phase 2 (SSD prefetch tier), Phase 3b (library API), Phase 4b (static schedule mode) — genuinely optional; mention as designed-but-not-yet-built in your writeup if you don't get to them. Interviewers respect a clear "here's what I'd add next" more than a rushed, buggy version of it.

## 6. Windows/WSL2 Development Note
- WSL2 + Docker Desktop (WSL2 backend) gives you a real Linux kernel — `perf`, `strace`, `mmap`, `pread` all genuinely work.
- Install the **Windows** NVIDIA driver (not a Linux driver inside WSL2) for GPU passthrough; verify with `docker run --gpus all nvidia/cuda:12.x-base nvidia-smi`.
- Check `io_uring` support on your WSL2 kernel version early (`wsl --update` for the latest) — if unsupported or flaky, plain `pread`-based async I/O is a safe fallback for the cold tier.
- Profiling (`nsys`/`perf`, especially GPU-side) can have reduced fidelity under WSL2; a native-Ubuntu cloud GPU box (RunPod/Lambda/Vast.ai) is a good fallback for profiling-heavy sessions.
