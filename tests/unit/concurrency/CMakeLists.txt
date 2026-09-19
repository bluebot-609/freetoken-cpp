add_executable(spsc_queue_test spsc_queue_test.cpp)
target_link_libraries(spsc_queue_test PRIVATE spsc_queue Threads::Threads GTest::gtest_main)
gtest_discover_tests(spsc_queue_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
