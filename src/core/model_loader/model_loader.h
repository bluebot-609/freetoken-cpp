#pragma once

#include <string>

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

}  // namespace freetoken::core
