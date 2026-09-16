#include <cstdio>

#include "ggml.h"

// Build/link smoke test only (dev_spec.md Section 8, step 1) — no engine logic yet.
int main() {
    std::printf("ggml version: %s\n", ggml_version());
    return 0;
}
