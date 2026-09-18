add_library(host_pool STATIC host_pool.cpp)
target_include_directories(host_pool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(gpu_cache STATIC gpu_cache.cpp)
target_include_directories(gpu_cache PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gpu_cache PUBLIC host_pool CUDA::cudart)
