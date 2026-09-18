#include "model_loader.h"

#include <cstdio>

#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

namespace freetoken::core {

bool load_gguf_and_run_forward_pass(const std::string& gguf_path) {
    // no_alloc=false: ggml eagerly reads each tensor's bytes via fread() into
    // data_ctx as part of loading (see third_party/ggml/src/gguf.cpp — the
    // reader uses fopen/fread throughout, not mmap; logged in
    // docs/citations.md per dev_spec.md Section 3.1's requirement to confirm
    // this explicitly). So unlike an mmap-based loader, there is no lazy
    // page-in to reason about here: the data is resident the moment this
    // call returns.
    ggml_context* data_ctx = nullptr;
    gguf_init_params gguf_params{ /*.no_alloc=*/ false, /*.ctx=*/ &data_ctx };
    gguf_context* gguf_ctx = gguf_init_from_file(gguf_path.c_str(), gguf_params);
    if (!gguf_ctx) {
        std::fprintf(stderr, "load_gguf_and_run_forward_pass: failed to open '%s'\n", gguf_path.c_str());
        return false;
    }

    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    std::printf("loaded %s: %lld tensor(s)\n", gguf_path.c_str(), static_cast<long long>(n_tensors));
    if (n_tensors == 0) {
        std::fprintf(stderr, "load_gguf_and_run_forward_pass: no tensors in '%s'\n", gguf_path.c_str());
        gguf_free(gguf_ctx);
        return false;
    }

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        const int64_t* ne = gguf_get_tensor_ne(gguf_ctx, i);
        std::printf("  [%lld] %s : %lldx%lldx%lldx%lld (%s)\n",
                    static_cast<long long>(i), name,
                    static_cast<long long>(ne[0]), static_cast<long long>(ne[1]),
                    static_cast<long long>(ne[2]), static_cast<long long>(ne[3]),
                    ggml_type_name(gguf_get_tensor_type(gguf_ctx, i)));
    }

    ggml_tensor* src = ggml_get_tensor(data_ctx, gguf_get_tensor_name(gguf_ctx, 0));

    // Deliberately trivial op (scale by 1.0) — the point is proving a real
    // ggml_cgraph runs end to end against loaded GGUF weights, not the op
    // itself. A real model's forward pass is Phase 1's job.
    ggml_init_params compute_params{ /*.mem_size=*/ 16 * 1024 * 1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ false };
    ggml_context* compute_ctx = ggml_init(compute_params);
    ggml_tensor* out = ggml_scale(compute_ctx, src, 1.0f);

    ggml_cgraph* graph = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(graph, out);

    const ggml_status status = ggml_graph_compute_with_ctx(compute_ctx, graph, /*n_threads=*/ 1);
    const bool ok = (status == GGML_STATUS_SUCCESS);
    if (ok) {
        // ggml_get_f32_1d() is mentioned only in comments in this ggml
        // version, not actually declared — read through the data pointer.
        std::printf("forward pass ok, output[0] = %f\n", static_cast<float*>(out->data)[0]);
    } else {
        std::fprintf(stderr, "load_gguf_and_run_forward_pass: graph compute failed (status=%d)\n", static_cast<int>(status));
    }

    ggml_free(compute_ctx);
    ggml_free(data_ctx);
    gguf_free(gguf_ctx);
    return ok;
}

}  // namespace freetoken::core
