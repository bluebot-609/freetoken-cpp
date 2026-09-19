# Building This Project

## What CMake actually is

CMake doesn't compile anything itself. It's a *build system generator*: you point it at a `CMakeLists.txt`, and it writes out actual build files (Makefiles, or Ninja files) for whatever tool your platform uses. That generated tool is what actually invokes the compiler. So there are always two steps:

1. **Configure** — CMake reads `CMakeLists.txt`, checks your compiler/CUDA/libraries exist, and generates build files into a separate directory (never inside `src/`).
2. **Build** — the generated Makefiles/Ninja actually compile and link.

You re-run configure only when a `CMakeLists.txt` changes (new source file, new option, new dependency). You re-run build every time you change `.cpp`/`.h` files — and it's incremental, only recompiling what changed.

## How this repo uses it

```
CMakeLists.txt              # root: sets C++17, enables GGML_CUDA, pulls in ggml + src/
├── third_party/ggml/       # vendored via git submodule, has its own CMakeLists.txt
└── src/CMakeLists.txt      # currently just adds src/cli/
    └── src/cli/CMakeLists.txt   # builds the freetoken-cli executable, links against ggml
```

The root `CMakeLists.txt` sets `GGML_CUDA ON` before pulling in `third_party/ggml`, which is why the build detects your GPU (RTX 4070 → CUDA arch `89`) and compiles ggml's CUDA backend alongside the CPU one. As new modules land (`src/core/model_loader`, `src/core/memory`, etc. — currently empty per `dev_spec.md` §1), each gets its own `CMakeLists.txt` that `src/CMakeLists.txt` will `add_subdirectory()`.

## Build artifacts — what gets produced, and where

Everything lands in the build directory (`build/`, gitignored — never commit these, they're regenerated from source):

| Artifact | Format | Where |
|---|---|---|
| Object files | `.o` | scattered under `build/CMakeFiles/.../*.o` — intermediate, never used directly |
| ggml static library | `libggml*.a` (or `.so` if `BUILD_SHARED_LIBS=ON`) | `build/third_party/ggml/src/` |
| This project's executables | ELF binary, no extension (e.g. `freetoken-cli`) | `build/src/cli/freetoken-cli` |
| Build metadata | `CMakeCache.txt`, `compile_commands.json` | `build/` root — the latter is useful for IDE tooling (clangd, etc.) |

There's no `.exe`/`.dll` here — this is a Linux/WSL2 build (see `roadmap.md` §6 and `dev_spec.md` §2 for why: `mlock`/`madvise`/`io_uring` don't exist on native Windows), so everything is ELF format.

## Commands

All commands run inside WSL2 Ubuntu (`wsl.exe -d Ubuntu -- bash -lc "..."` from Windows, or just directly if you're already in a WSL shell). Run from the repo root.

**First-time configure** (creates `build/`):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```
- `-S .` — source directory (this repo root, where the root `CMakeLists.txt` is)
- `-B build` — where to put generated build files (this is what "the build directory" means everywhere in this doc)
- `-DCMAKE_BUILD_TYPE=Release` — optimized build. Use `Debug` instead when you need working `gdb`/debug symbols.

**Build** (compile everything; safe to re-run anytime, only rebuilds what changed):
```bash
cmake --build build -j$(nproc)
```
- `-j$(nproc)` — parallel jobs, one per CPU core. Drop it (`cmake --build build`) to build single-threaded if you want quieter/serial output while debugging a build error.

**Build just one target** (e.g. only the CLI, skip rebuilding all of ggml if it's already built):
```bash
cmake --build build --target freetoken-cli
```

**Clean** (delete compiled output, keep the configured `CMakeCache.txt`/generated files):
```bash
cmake --build build --target clean
```

**Full reset** (when something's actually stale/broken, not just "I want a rebuild" — e.g. after changing `GGML_CUDA` or another CMake *option*, or if you suspect an incremental-build staleness issue per `docs/citations.md`):
```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**Run the built binary:**
```bash
./build/src/cli/freetoken-cli
```

## Testing

No framework (Catch2/GoogleTest) is wired in yet — `dev_spec.md` §6 names them as the eventual choice, pulled via CMake `FetchContent` when that happens. For now, every test is a plain executable that prints `PASS`/`FAIL` per check and returns a real process exit code (0 = all passed) — same idea as a framework, just hand-rolled, in `tests/unit/` (pure-logic checks) and `tests/integration/` (real GPU/pool/cache/model checks). Build and run any one directly:
```bash
cmake --build build --target spsc_queue_test
./build/tests/unit/spsc_queue_test
```

**Run everything at once**, with a consolidated report (real numbers — bandwidths, timings, speedups — not just pass/fail) saved to `benchmark/reports/`:
```bash
bash scripts/run_benchmarks.sh
```

## Generating API docs (Doxygen)

Combines the API reference (extracted from `///` comments in `src/`'s headers — plain `//` comments are NOT picked up, that's a Doxygen convention, not a bug), the project's markdown docs, and the test tree into one browsable HTML site:
```bash
doxygen Doxyfile
```
Output: `docs_site/html/index.html` (open it in a browser). Gitignored — fully reproducible from source, so it isn't committed. Regenerate any time the code or docs change; nothing here is hand-maintained.
