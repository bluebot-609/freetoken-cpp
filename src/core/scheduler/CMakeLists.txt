add_library(calibration STATIC calibration.cpp)
target_include_directories(calibration PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(calibration PUBLIC CUDA::cudart ggml)
