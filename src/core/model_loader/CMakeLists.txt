add_library(model_loader STATIC model_loader.cpp)
target_include_directories(model_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(model_loader PUBLIC ggml)
