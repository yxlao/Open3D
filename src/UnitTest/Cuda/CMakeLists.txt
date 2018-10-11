add_executable(TestContainerCuda Container/TestContainerCuda.cpp)
target_link_libraries(TestContainerCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestImageCuda Geometry/TestImageCuda.cpp)
target_link_libraries(TestImageCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestTransformCuda Geometry/TestTransformCuda.cpp)
target_link_libraries(TestTransformCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestReduction2DCuda Odometry/TestReduction2DCuda.cpp)
target_link_libraries(TestReduction2DCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestRGBDOdometryCuda Odometry/TestRGBDOdometryCuda.cpp)
target_link_libraries(TestRGBDOdometryCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestUniformTSDFVolumeCuda Integration/TestUniformTSDFVolumeCuda.cpp)
target_link_libraries(TestUniformTSDFVolumeCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)