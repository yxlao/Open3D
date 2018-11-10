add_executable(TestScalableTSDFVolumeCuda TestScalableTSDFVolumeCuda.cpp)
target_link_libraries(TestScalableTSDFVolumeCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestUniformTSDFVolumeCuda TestUniformTSDFVolumeCuda.cpp)
target_link_libraries(TestUniformTSDFVolumeCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)