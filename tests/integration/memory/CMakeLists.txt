add_executable(memory_smoke memory_smoke.cpp)
target_link_libraries(memory_smoke PRIVATE host_pool gpu_cache CUDA::cudart GTest::gtest_main)
gtest_discover_tests(memory_smoke WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
