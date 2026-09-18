cmake_minimum_required(VERSION 3.24)
project(freetoken-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# NVIDIA-only scope for v1 (dev_spec.md Section 2, Section 9 non-goals).
option(GGML_CUDA "ggml: use CUDA" ON)

# Found here, not just left to ggml's internal find_package(CUDAToolkit)
# call, because our own code (src/core/memory/gpu_cache.cpp) calls CUDA
# runtime functions directly too — don't rely on another subdirectory's
# side effect for something we need ourselves.
find_package(CUDAToolkit REQUIRED)

add_subdirectory(third_party/ggml)

add_subdirectory(src)
add_subdirectory(tests/fixtures)
add_subdirectory(tests/integration)
