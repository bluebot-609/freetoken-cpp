add_executable(freetoken-cli main.cpp)
target_link_libraries(freetoken-cli PRIVATE ggml model_loader)
