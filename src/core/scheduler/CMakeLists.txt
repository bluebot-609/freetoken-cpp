add_library(calibration STATIC calibration.cpp)
target_include_directories(calibration PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(calibration PUBLIC CUDA::cudart ggml)

add_library(qstar_scheduler STATIC qstar_scheduler.cpp selection_policy.cpp)
target_include_directories(qstar_scheduler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qstar_scheduler PUBLIC calibration host_pool gpu_cache spsc_queue ggml Threads::Threads)
