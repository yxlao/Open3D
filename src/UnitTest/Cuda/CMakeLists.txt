# Containers
add_executable(TestHashTableCuda Container/TestHashTableCuda.cpp)
target_link_libraries(TestHashTableCuda ${CMAKE_PROJECT_NAME}
        ${3RDPARTY_LIBRARIES}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestArrayCuda Container/TestArrayCuda.cpp)
target_link_libraries(TestArrayCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestLinkedListCuda Container/TestLinkedListCuda.cpp)
target_link_libraries(TestLinkedListCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

# Geometry
add_executable(TestImageCuda Geometry/TestImageCuda.cpp)
target_link_libraries(TestImageCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestTransformCuda Geometry/TestTransformCuda.cpp)
target_link_libraries(TestTransformCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

# Odometry
add_executable(TestReduction2DCuda Odometry/TestReduction2DCuda.cpp)
target_link_libraries(TestReduction2DCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestRGBDOdometryCuda Odometry/TestRGBDOdometryCuda.cpp)
target_link_libraries(TestRGBDOdometryCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

# Integration
add_executable(TestUniformTSDFVolumeCuda Integration/TestUniformTSDFVolumeCuda.cpp)
target_link_libraries(TestUniformTSDFVolumeCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestUniformMeshVolumeCuda Integration/TestUniformMeshVolumeCuda.cpp)
target_link_libraries(TestUniformMeshVolumeCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestScalableTSDFVolumeCuda Integration/TestScalableTSDFVolumeCuda.cpp)
target_link_libraries(TestScalableTSDFVolumeCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestScalableMeshVolumeCuda Integration/TestScalableMeshVolumeCuda.cpp)
target_link_libraries(TestScalableMeshVolumeCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)