#include <cstdio>

#include "ggml.h"
#include "model_loader.h"

int main(int argc, char** argv) {
    std::printf("ggml version: %s\n", ggml_version());

    // dev_spec.md Section 8, step 2 (Phase 0 exit criterion): load a small
    // GGUF file and run one real forward pass, no caching/scheduler yet.
    // Optional so the plain link smoke test (step 1) keeps working with no
    // arguments.
    if (argc > 1) {
        return freetoken::core::load_gguf_and_run_forward_pass(argv[1]) ? 0 : 1;
    }
    return 0;
}
