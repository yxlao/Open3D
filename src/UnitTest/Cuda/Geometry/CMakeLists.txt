# Geometry
add_executable(TestImageCuda TestImageCuda.cpp)
target_link_libraries(TestImageCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestTransformCuda TestTransformCuda.cpp)
target_link_libraries(TestTransformCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestTriangleMeshCuda TestTriangleMeshCuda.cpp)
target_link_libraries(TestTriangleMeshCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)