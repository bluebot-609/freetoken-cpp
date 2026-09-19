#pragma once

#include <string>

#include "host_pool.h"

namespace freetoken::core {

// Loads a GGUF file (reusing ggml's own reader — dev_spec.md Section 3.1,
// do not reimplement) and runs one real ggml_cgraph forward pass over the
// first tensor found, purely to prove the load -> graph -> compute path
// end to end. No caching, no CPU/GPU split scheduler: that's Phase 1
// (Module 3.4), deliberately out of scope here.
//
// Exposed parameter: model path (dev_spec.md Section 3.1 — context length,
// batch size, and quantization format are read from GGUF metadata once a
// real model is loaded, not applicable to this smoke-test path yet).
bool load_gguf_and_run_forward_pass(const std::string& gguf_path);

// Loads real MoE expert weight data from a real GGUF model into `pool`.
//
// NOT REAL INFERENCE DATA -- see PROGRESS.md's placeholder ledger. A real
// MoE expert needs three weight matrices (gate/up/down, the SwiGLU MLP
// shape). This extracts only `blk.N.ffn_gate_exps.weight`, sliced per
// expert, as representative real weight BYTES -- enough to exercise the
// Host-Resident Pool / GPU-Expert Cache / q* scheduler pipeline against
// genuine model data instead of synthetic filler, but NOT enough to run a
// real forward pass. That requires fetching gate+up+down together and is
// not built.
//
// GGUF layout assumption (verified against a real downloaded model, not
// guessed): each layer's expert weights are stored as ONE merged 3D tensor
// (not one tensor per expert), with the expert index as the outermost
// dimension (ne[2]). Since ggml tensors are stored contiguously, expert
// e's data is the contiguous byte range
// [e * (tensor_size / n_expert), (e+1) * (tensor_size / n_expert)).
//
// Layer count is discovered by scanning for blk.0, blk.1, ... until a
// layer's ffn_gate_exps tensor isn't found -- NOT read from a GGUF
// metadata key (e.g. "<arch>.block_count"), which would be the more
// correct approach once this needs to generalize beyond one test model.
//
// expert_id scheme: layer * experts_per_layer + expert_index.
// Returns the number of (layer, expert) pairs registered, or 0 on failure.
int load_moe_experts_into_pool(const std::string& gguf_path, HostResidentPool& pool);

}  // namespace freetoken::core
