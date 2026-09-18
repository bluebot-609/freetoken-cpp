// One-off generator for tests/fixtures/hello.gguf — a minimal, valid GGUF
// file (one named F32 tensor, 4 elements) used to exercise the GGUF-load ->
// ggml_cgraph -> compute path in src/core/model_loader without depending on
// downloading a real model (dev_spec.md Section 6: fixtures should be small
// test tensors, not full checkpoints).
//
// Run once (from the build dir) to (re)produce the fixture:
//   ./tests/fixtures/make_hello_gguf tests/fixtures/hello.gguf
#include <cstdio>

#include "ggml.h"
#include "gguf.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output.gguf>\n", argv[0]);
        return 1;
    }

    ggml_init_params params{ /*.mem_size=*/ ggml_tensor_overhead() + 256, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ false };
    ggml_context* ctx = ggml_init(params);

    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_set_name(t, "hello.weight");
    // ggml_set_f32_1d() is mentioned only in comments in this ggml version,
    // not actually declared — write through the data pointer directly
    // (valid since ggml_init() above used no_alloc=false).
    float* data = static_cast<float*>(t->data);
    for (int i = 0; i < 8; ++i) {
        data[i] = static_cast<float>(i + 5);
    }

    gguf_context* gguf_ctx = gguf_init_empty();
    gguf_add_tensor(gguf_ctx, t);
    gguf_write_to_file(gguf_ctx, argv[1], /*only_meta=*/ false);

    gguf_free(gguf_ctx);
    ggml_free(ctx);

    std::printf("wrote %s\n", argv[1]);
    return 0;
}
