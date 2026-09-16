# Progress Log

## Current Phase
Phase 0 — Foundations (per roadmap.md). Step 1 of dev_spec.md §8 build order complete.

## Last Updated
2026-09-17 — repo scaffolded, ggml submodule pinned, CMake build confirmed working end-to-end on WSL2 Ubuntu with CUDA backend.

## Completed
- Environment: WSL2 Ubuntu 26.04 LTS installed and set as the dev environment (native Windows lacks `mlock`/`madvise`/`io_uring`). GPU passthrough confirmed (`nvidia-smi` works in WSL2). Toolchain installed: `build-essential`, `cmake` 4.2.3, `nvidia-cuda-toolkit` (nvcc 12.4.131 — driver is 610.88/CUDA 13.3-capable, so 12.4 toolkit is comfortably compatible), `strace`, `perf`.
- Repo structure scaffolded per dev_spec.md §1 (`src/core/{model_loader,memory,scheduler,concurrency}`, `src/server`, `src/harness`, `src/cli`, `benchmark/{scenarios,reports}`, `tests/{unit,integration,fixtures}`, `scripts/`, `docs/`).
- `third_party/ggml` vendored as a git submodule, pinned to commit `456172ec733a135778adcd32d00e576a58232e45` (not tracking a branch).
- Root `CMakeLists.txt` + `src/CMakeLists.txt` + `src/cli/CMakeLists.txt` + `src/cli/main.cpp` (link smoke test only — calls `ggml_version()`, no engine logic).
- **Module 0 (build scaffold):** configure + build + run all verified on WSL2. Detected CUDA arch `89` (RTX 4070) correctly; CUDA backend included in the ggml build. Binary at `~/build/freetoken-cpp/src/cli/freetoken-cli` runs and prints `ggml version: 0.24.0`.
- `docs/citations.md` created; two entries logged so far (ggml pin, WSL2 build-location gotcha).
- `dev_spec.md` §10 and `roadmap.md` Phase 5b added: Docker-based distribution deferred to post-v1, with the specific `--gpus`/`memlock`/`seccomp` gotchas documented so they aren't rediscovered later.

## In Progress
Nothing mid-flight. Ready to start Phase 0's remaining exit criterion: a real ggml "hello world" — load a small GGUF model and run a forward pass, no custom caching yet.

## Next Concrete Step
Phase 0 exit criterion (roadmap.md): write a small program (extend `src/cli/main.cpp` or add a new one under `src/core/model_loader/`) that loads a small quantized GGUF model via ggml and runs a forward pass with no custom caching/scheduler. Need a small test GGUF fixture first — either download a tiny model or use one of ggml's own test fixtures/examples as a reference for the loading API (`ggml_init`, `gguf_init_from_file`, building a `ggml_cgraph`). Look at `third_party/ggml/examples/` for a minimal working example to adapt.

Also outstanding, not yet done: explicitly confirm/document whether ggml's GGUF loader uses `mmap` or `read()` (dev_spec.md §3.1 requires this be documented, not assumed) — check `third_party/ggml/src/gguf.cpp` once we get there.

## Open Questions / Flags for the Author
- Build directory MUST stay on WSL2's native Linux filesystem (e.g. `~/build/freetoken-cpp`), NOT under `/mnt/c/...` — CMake's `try_compile` (used for OpenMP and CUDA compiler detection) fails with "Operation not permitted" when the build dir is on the Windows-mounted DrvFs. Source can stay on `/mnt/c` (that's fine, only read access needed); only the build output directory needs to be native-fs. Documented in `docs/citations.md`.
- `perf`/hardware-counter fidelity under WSL2's custom kernel (`6.6.87.2-microsoft-standard-WSL2`) is unverified yet — flagged in `roadmap.md` §6 as a known risk; will confirm at the first profiling checkpoint (dev_spec.md §8 step 2).
- No FreeToken reference repo has been cloned/read yet (recommended as read-only reference, not vendored) — should happen before writing the q* scheduler (Phase 1), not blocking Phase 0.
