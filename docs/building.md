# Building This Project

## What CMake actually is

CMake doesn't compile anything itself. It's a *build system generator*: you point it at a `CMakeLists.txt`, and it writes out actual build files (Makefiles, or Ninja files) for whatever tool your platform uses. That generated tool is what actually invokes the compiler. So there are always two steps:

1. **Configure** — CMake reads `CMakeLists.txt`, checks your compiler/CUDA/libraries exist, and generates build files into a separate directory (never inside `src/`).
2. **Build** — the generated Makefiles/Ninja actually compile and link.

You re-run configure only when a `CMakeLists.txt` changes (new source file, new option, new dependency). You re-run build every time you change `.cpp`/`.h` files — and it's incremental, only recompiling what changed.

## How this repo uses it

```
CMakeLists.txt                    # root: sets C++17, enables GGML_CUDA, fetches GoogleTest, pulls in ggml + src/ + tests/
├── third_party/ggml/             # vendored via git submodule, has its own CMakeLists.txt
├── src/CMakeLists.txt            # adds core/ and cli/
│   ├── core/CMakeLists.txt       # adds model_loader/, memory/, concurrency/, scheduler/ -- one library target each
│   └── cli/CMakeLists.txt        # builds the freetoken-cli executable
└── tests/
    ├── unit/<module>/            # pure-logic tests, one CMakeLists.txt per module (mirrors src/core/<module>/)
    ├── integration/<module>/     # real GPU/pool/cache/model tests, same mirroring
    └── fixtures/                 # small test-data generators (e.g. tests/fixtures/hello.gguf)
```

The root `CMakeLists.txt` sets `GGML_CUDA ON` before pulling in `third_party/ggml`, which is why the build detects your GPU (RTX 4070 → CUDA arch `89`) and compiles ggml's CUDA backend alongside the CPU one.

## Build artifacts — what gets produced, and where

Everything lands in the build directory (`build/`, gitignored — never commit these, they're regenerated from source):

| Artifact | Format | Where |
|---|---|---|
| Object files | `.o` | scattered under `build/CMakeFiles/.../*.o` — intermediate, never used directly |
| ggml static library | `libggml*.a` (or `.so` if `BUILD_SHARED_LIBS=ON`) | `build/third_party/ggml/src/` |
| This project's own libraries | `.a` static libs, one per `src/core/<module>` | `build/src/core/<module>/` |
| This project's executables | ELF binary, no extension (e.g. `freetoken-cli`) | `build/src/cli/freetoken-cli` |
| Test executables | ELF binary, one per `tests/<unit\|integration>/<module>/*.cpp` | `build/tests/<unit\|integration>/<module>/` |
| Build metadata | `CMakeCache.txt`, `compile_commands.json` | `build/` root — the latter is useful for IDE tooling (clangd, etc.) |

There's no `.exe`/`.dll` here — this is a Linux/WSL2 build (see `docs/roadmap.md` §6 and `docs/dev_spec.md` §2 for why: `mlock`/`madvise`/`io_uring` don't exist on native Windows), so everything is ELF format.

## Commands

All commands run inside WSL2 Ubuntu (`wsl.exe -d Ubuntu -- bash -lc "..."` from Windows, or just directly if you're already in a WSL shell). Run from the repo root.

**First-time configure** (creates `build/`; also fetches GoogleTest the first time, so it's slower once):
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

Every test is a real [GoogleTest](https://github.com/google/googletest) case (fetched via CMake `FetchContent`, pinned to `v1.15.2` — per this project's own "no package manager beyond FetchContent/submodules" rule), registered with CTest via `gtest_discover_tests()`. Tests live in `tests/unit/<module>/` (pure logic, no GPU) and `tests/integration/<module>/` (real GPU/pool/cache/model checks), mirroring `src/core/<module>/` — a test's folder tells you what it covers.

Each test has a real, stable ID: `TestSuite.TestName` (e.g. `QStarScheduler.FasterThanNaiveStallAndCopyBaseline`).

**Run everything:**
```bash
ctest --test-dir build
```

**Run one suite** (regex match against test IDs):
```bash
ctest --test-dir build -R QStarScheduler
```

**See what each test actually printed** (bandwidths, timings, speedups — CTest hides stdout on success by default):
```bash
ctest --test-dir build -V
```

**Build and run one test binary directly** (bypasses CTest, useful for GoogleTest's own flags like `--gtest_filter` or `--gtest_repeat`):
```bash
cmake --build build --target spsc_queue_test
./build/tests/unit/concurrency/spsc_queue_test --gtest_repeat=8
```

**Run everything at once with a consolidated numbers summary**, saved to `benchmark/reports/`:
```bash
bash scripts/run_benchmarks.sh
```

## Generating API docs (Doxygen)

Combines the API reference (extracted from `///` comments in `src/`'s headers — plain `//` comments are NOT picked up, that's a Doxygen convention, not a bug), every markdown doc, and the test tree into one browsable HTML site:
```bash
bash scripts/build_docs.sh
```
Output: `docs_site/html/index.html`, plus a redirect page at the repo root (`index.html`) that opens straight into it. Both are gitignored — fully reproducible from source, so they aren't committed. Regenerate any time the code or docs change; nothing in `docs_site/` is hand-maintained. (Equivalent to running `doxygen Doxyfile` directly, if you don't need the root redirect.)
