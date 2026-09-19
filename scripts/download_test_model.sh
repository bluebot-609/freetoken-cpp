#!/usr/bin/env bash
# Downloads the tiny real MoE GGUF test model used by
# tests/integration/qstar_scheduler_smoke.cpp (Part 4).
#
# NOT random filler data -- a real MoE architecture (6 layers, 2 experts/
# layer) with random (untrained) weights, small enough to keep in fast CI
# loops. See model_loader.h's load_moe_experts_into_pool() and
# docs/citations.md for the full layout writeup.
#
# Uses the `hf` CLI, not curl/wget: this repo's storage backend is
# Hugging Face's Xet system, and a plain HTTP GET on it has been observed
# to silently return a zero-byte-filled file of the correct size and a
# 200 status -- not an error, just wrong data. Verify actual byte content
# after any manual download (e.g. `xxd file | head` and check for the
# `GGUF` magic bytes), never trust HTTP status + file size alone.
set -euo pipefail

if ! command -v hf >/dev/null 2>&1; then
    echo "error: 'hf' CLI not found. Install it first:" >&2
    echo "  curl -LsSf https://hf.co/cli/install.sh | bash -s" >&2
    echo "(needs python3-venv: sudo apt-get install -y python3-venv)" >&2
    exit 1
fi

cd "$(dirname "$0")/.."
mkdir -p models
hf download mradermacher/tiny-random-granite-moe-GGUF tiny-random-granite-moe.f16.gguf --local-dir models
echo "downloaded to models/tiny-random-granite-moe.f16.gguf"
