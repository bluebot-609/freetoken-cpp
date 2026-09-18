# Progress Log

## Current Phase
Phase 0 — Foundations (per roadmap.md). dev_spec.md §8 steps 1 AND 2 complete. Phase 0's exit criterion is met; ready to start Phase 1.

## Last Updated
2026-09-19 — Module 3.1 (Model Loader) started: loads a real GGUF file via ggml's own reader and runs one real ggml_cgraph forward pass. mmap-vs-read() question (dev_spec.md §3.1) answered and verified with strace: it's read().

## Completed
- Environment: WSL2 Ubuntu 26.04 LTS installed and set as the dev environment (native Windows lacks `mlock`/`madvise`/`io_uring`). GPU passthrough confirmed (`nvidia-smi` works in WSL2). Toolchain installed: `build-essential`, `cmake` 4.2.3, `nvidia-cuda-toolkit` (nvcc 12.4.131 — driver is 610.88/CUDA 13.3-capable, so 12.4 toolkit is comfortably compatible), `strace`, `perf`.
- Repo structure scaffolded per dev_spec.md §1 (`src/core/{model_loader,memory,scheduler,concurrency}`, `src/server`, `src/harness`, `src/cli`, `benchmark/{scenarios,reports}`, `tests/{unit,integration,fixtures}`, `scripts/`, `docs/`).
- `third_party/ggml` vendored as a git submodule, pinned to commit `456172ec733a135778adcd32d00e576a58232e45` (not tracking a branch).
- Root `CMakeLists.txt` + `src/CMakeLists.txt` + `src/cli/CMakeLists.txt` + `src/cli/main.cpp` (link smoke test only — calls `ggml_version()`, no engine logic).
- **Module 0 (build scaffold):** configure + build + run all verified on WSL2, in-repo. Detected CUDA arch `89` (RTX 4070) correctly; CUDA backend included in the ggml build. Binary at `build/src/cli/freetoken-cli` (repo-relative) runs and prints `ggml version: 0.24.0`.
- `docs/citations.md` created and corrected (see Open Questions below — the original "build dir must be on native fs" finding was wrong).
- `docs/building.md` added: CMake primer, what this project's CMakeLists.txt files do, build artifact formats/locations, and the configure/build/clean/rebuild/run command reference. Testing section is a placeholder — no test framework wired in yet.
- `dev_spec.md` §10 and `roadmap.md` Phase 5b added: Docker-based distribution deferred to post-v1, with the specific `--gpus`/`memlock`/`seccomp` gotchas documented so they aren't rediscovered later.
- **Module 3.1 (Model Loader), first real implementation:** `src/core/model_loader/model_loader.{h,cpp}` — `load_gguf_and_run_forward_pass()` loads a GGUF file via `gguf_init_from_file` (ggml's own reader, not reimplemented) and runs one real `ggml_cgraph` (scale-by-1.0 op) via `ggml_graph_compute_with_ctx`. Deliberately trivial op — proving the load→graph→compute path, not real model math (that's Phase 1). No caching/scheduler yet, as required.
- `tests/fixtures/make_hello_gguf.cpp` — one-off generator producing `tests/fixtures/hello.gguf` (128 bytes, one named F32 tensor `hello.weight`, 4 elements `[1,2,3,4]`), since no tiny GGUF fixture existed anywhere in the ggml submodule or examples. Committed the generator (reusable for future fixtures) and the small generated `.gguf` file itself.
- **mmap vs read() answered** (dev_spec.md §3.1 required this be confirmed, not assumed): ggml's GGUF reader uses `fopen`/`fread`, i.e. plain `read()` — no `mmap`. Verified two ways: reading `gguf.cpp` source, and `strace -e trace=openat,mmap,read` on the actual run, which shows `openat(hello.gguf)` followed by exactly one `read()`, no `mmap` on that fd. Full detail + the Phase 1 consequence (no lazy page-in to reason about) logged in `docs/citations.md`.
- End-to-end verified: `./build/src/cli/freetoken-cli tests/fixtures/hello.gguf` → loads the tensor, prints its shape/type, runs the forward pass, prints `output[0] = 1.000000` (correct: 1×1.0).
- Discovered along the way: `ggml_get_f32_1d`/`ggml_set_f32_1d` are mentioned only in comments in this ggml version (0.24.0), not actually declared/exported — used direct `tensor->data` pointer access instead in both the fixture generator and the loader.

## In Progress
Nothing mid-flight. Phase 0 is done. Ready to start Phase 1 (dev_spec.md §8 steps 3+): Host-Resident Pool (3.2) and GPU-Expert Cache (3.3) with a naive synchronous stall-and-copy fallback as the baseline.

## Next Concrete Step
Read the FreeToken paper's q* section and skim the `FlashML-org/FreeToken` repo (read-only reference, per roadmap.md §0 — do not vendor it) before starting Phase 1, since the paper's actual formulation should drive Module 3.2/3.3's design rather than guessing. Then, per dev_spec.md §8 step 3 (the correct next step — do NOT jump ahead to step 4/5): implement the Host-Resident Pool (3.2) and GPU-Expert Cache (3.3) with a naive synchronous stall-and-copy fallback. This is deliberately the control/baseline the rest of Phase 1 gets measured against — build it before the SPSC queue (3.8, step 4) or the q* scheduler (3.4, step 5).

## Open Questions / Flags for the Author
- CORRECTION to a prior entry: the build dir does NOT need to be on native WSL2 filesystem — re-tested and an in-repo `build/` (on `/mnt/c/...`) configures and builds fine. The original "Operation not permitted" `try_compile` failure was a one-off (likely Defender/AV momentarily locking a freshly-written `.exe`), not a structural DrvFs limitation. Build artifacts now live at `build/` in the repo, gitignored. See `docs/citations.md` for the corrected entry.
- Minor known gremlin: running the built binary via `wsl.exe -d Ubuntu -- <bare /mnt/c/... path>` from Windows-side Git Bash gets mangled by MSYS2's automatic path conversion (turns it into a bogus `C:/Program Files/Git/mnt/c/...` path). Workaround: wrap in `bash -lc "cd / && /mnt/c/.../binary"` so the argument isn't a bare leading-slash token. Not a project bug, just a Git-Bash-on-Windows quirk when driving WSL2 from these tools.
- `perf`/hardware-counter fidelity under WSL2's custom kernel (`6.6.87.2-microsoft-standard-WSL2`) is unverified yet — flagged in `roadmap.md` §6 as a known risk; will confirm at the first profiling checkpoint (dev_spec.md §8 step 2).
- No FreeToken reference repo has been cloned/read yet (recommended as read-only reference, not vendored) — should happen before writing the q* scheduler (Phase 1), not blocking Phase 0.
