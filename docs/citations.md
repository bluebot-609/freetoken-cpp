# Citations Log

Running log of every external source referenced during implementation. Format:
`[date] [module] [source] — [what was learned/used]`

---

[2026-09-17] [build/third_party] [ggml-org/ggml @ 456172ec733a135778adcd32d00e576a58232e45] — Vendored as a git submodule, pinned to this commit SHA (not tracking a branch) per dev_spec.md §2 pinning discipline. Repo tag context: "ggml : bump version to 0.24.0 (#1627)".

[2026-09-17] [build/CMake] [observed while configuring on WSL2] — CMake's `try_compile` (used by ggml's own OpenMP and CUDA-compiler-ABI detection) failed once with "Operation not permitted" when the build directory was on `/mnt/c/...` (Windows filesystem via DrvFs). At the time this looked like a hard DrvFs limitation, so the build was moved to `~/build/freetoken-cpp` on native ext4.

[2026-09-19] [build/CMake] [correction to the entry above] — Re-tested: configure + full build (`cmake -S . -B build && cmake --build build`) both succeed with the build directory in-repo on `/mnt/c/Development/freetoken-cpp/build`. The original failure was a one-off (most likely Windows Defender/antivirus briefly locking a freshly-written `.exe` during `try_compile`, not a real DrvFs/9p limitation). Build artifacts now live in-repo at `build/` (gitignored) as the author wants them visible there. If `try_compile`-style "Operation not permitted" errors recur, retry once before assuming it's structural — see `docs/building.md`.
