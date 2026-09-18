add_executable(memory_smoke memory_smoke.cpp)
target_link_libraries(memory_smoke PRIVATE host_pool gpu_cache CUDA::cudart)
