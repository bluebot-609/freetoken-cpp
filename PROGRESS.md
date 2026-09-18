# Progress Log

## Current Phase
Phase 0 — Foundations (per roadmap.md). Step 1 of dev_spec.md §8 build order complete.

## Last Updated
2026-09-19 — build directory moved back in-repo (`build/`, gitignored); confirmed working. Added `docs/building.md` (CMake primer + command reference).

## Completed
- Environment: WSL2 Ubuntu 26.04 LTS installed and set as the dev environment (native Windows lacks `mlock`/`madvise`/`io_uring`). GPU passthrough confirmed (`nvidia-smi` works in WSL2). Toolchain installed: `build-essential`, `cmake` 4.2.3, `nvidia-cuda-toolkit` (nvcc 12.4.131 — driver is 610.88/CUDA 13.3-capable, so 12.4 toolkit is comfortably compatible), `strace`, `perf`.
- Repo structure scaffolded per dev_spec.md §1 (`src/core/{model_loader,memory,scheduler,concurrency}`, `src/server`, `src/harness`, `src/cli`, `benchmark/{scenarios,reports}`, `tests/{unit,integration,fixtures}`, `scripts/`, `docs/`).
- `third_party/ggml` vendored as a git submodule, pinned to commit `456172ec733a135778adcd32d00e576a58232e45` (not tracking a branch).
- Root `CMakeLists.txt` + `src/CMakeLists.txt` + `src/cli/CMakeLists.txt` + `src/cli/main.cpp` (link smoke test only — calls `ggml_version()`, no engine logic).
- **Module 0 (build scaffold):** configure + build + run all verified on WSL2, in-repo. Detected CUDA arch `89` (RTX 4070) correctly; CUDA backend included in the ggml build. Binary at `build/src/cli/freetoken-cli` (repo-relative) runs and prints `ggml version: 0.24.0`.
- `docs/citations.md` created and corrected (see Open Questions below — the original "build dir must be on native fs" finding was wrong).
- `docs/building.md` added: CMake primer, what this project's CMakeLists.txt files do, build artifact formats/locations, and the configure/build/clean/rebuild/run command reference. Testing section is a placeholder — no test framework wired in yet.
- `dev_spec.md` §10 and `roadmap.md` Phase 5b added: Docker-based distribution deferred to post-v1, with the specific `--gpus`/`memlock`/`seccomp` gotchas documented so they aren't rediscovered later.

## In Progress
Nothing mid-flight. Ready to start Phase 0's remaining exit criterion: a real ggml "hello world" — load a small GGUF model and run a forward pass, no custom caching yet.

## Next Concrete Step
Phase 0 exit criterion (roadmap.md): write a small program (extend `src/cli/main.cpp` or add a new one under `src/core/model_loader/`) that loads a small quantized GGUF model via ggml and runs a forward pass with no custom caching/scheduler. Need a small test GGUF fixture first — either download a tiny model or use one of ggml's own test fixtures/examples as a reference for the loading API (`ggml_init`, `gguf_init_from_file`, building a `ggml_cgraph`). Look at `third_party/ggml/examples/` for a minimal working example to adapt.

Also outstanding, not yet done: explicitly confirm/document whether ggml's GGUF loader uses `mmap` or `read()` (dev_spec.md §3.1 requires this be documented, not assumed) — check `third_party/ggml/src/gguf.cpp` once we get there.

## Open Questions / Flags for the Author
- CORRECTION to a prior entry: the build dir does NOT need to be on native WSL2 filesystem — re-tested and an in-repo `build/` (on `/mnt/c/...`) configures and builds fine. The original "Operation not permitted" `try_compile` failure was a one-off (likely Defender/AV momentarily locking a freshly-written `.exe`), not a structural DrvFs limitation. Build artifacts now live at `build/` in the repo, gitignored. See `docs/citations.md` for the corrected entry.
- Minor known gremlin: running the built binary via `wsl.exe -d Ubuntu -- <bare /mnt/c/... path>` from Windows-side Git Bash gets mangled by MSYS2's automatic path conversion (turns it into a bogus `C:/Program Files/Git/mnt/c/...` path). Workaround: wrap in `bash -lc "cd / && /mnt/c/.../binary"` so the argument isn't a bare leading-slash token. Not a project bug, just a Git-Bash-on-Windows quirk when driving WSL2 from these tools.
- `perf`/hardware-counter fidelity under WSL2's custom kernel (`6.6.87.2-microsoft-standard-WSL2`) is unverified yet — flagged in `roadmap.md` §6 as a known risk; will confirm at the first profiling checkpoint (dev_spec.md §8 step 2).
- No FreeToken reference repo has been cloned/read yet (recommended as read-only reference, not vendored) — should happen before writing the q* scheduler (Phase 1), not blocking Phase 0.
