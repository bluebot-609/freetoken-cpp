# spsc_queue.h is header-only (template class) -- nothing to compile into a
# library here. This file exists so other CMakeLists.txt can still
# target_include_directories against this directory consistently.
add_library(spsc_queue INTERFACE)
target_include_directories(spsc_queue INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
