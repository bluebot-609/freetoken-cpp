# Citations Log

Running log of every external source referenced during implementation. Format:
`[date] [module] [source] — [what was learned/used]`

---

[2026-09-17] [build/third_party] [ggml-org/ggml @ 456172ec733a135778adcd32d00e576a58232e45] — Vendored as a git submodule, pinned to this commit SHA (not tracking a branch) per dev_spec.md §2 pinning discipline. Repo tag context: "ggml : bump version to 0.24.0 (#1627)".

[2026-09-17] [build/CMake] [observed while configuring on WSL2] — CMake's `try_compile` (used by ggml's own OpenMP and CUDA-compiler-ABI detection) fails with "Operation not permitted" when the build directory lives on `/mnt/c/...` (Windows filesystem via DrvFs). Fix: point `-B` at a directory on WSL2's native ext4 filesystem (e.g. `~/build/freetoken-cpp`); `-S` (source) can stay on `/mnt/c` without issue since it's read-only for the configure/build step. Not a CUDA or toolchain bug — a DrvFs/9p filesystem limitation with executable file semantics.
