# Visualizer
add_executable(TestVisualizerCuda TestVisualizerCuda.cpp)
target_link_libraries(TestVisualizerCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)