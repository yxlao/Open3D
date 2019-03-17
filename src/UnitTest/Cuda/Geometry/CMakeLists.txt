# Geometry
include_directories(/usr/local/cuda/include)

add_executable(TestImageCuda TestImageCuda.cpp)
target_link_libraries(TestImageCuda ${CMAKE_PROJECT_NAME} Cuda
        ${GTEST_LIBRARIES} pthread)

#add_executable(TestRGBDImageCuda TestRGBDImageCuda.cpp)
#target_link_libraries(TestRGBDImageCuda ${CMAKE_PROJECT_NAME}
#        ${GTEST_LIBRARIES} pthread)
#
#add_executable(TestTransformCuda TestTransformCuda.cpp)
#target_link_libraries(TestTransformCuda ${CMAKE_PROJECT_NAME}
#        ${GTEST_LIBRARIES} pthread)
#
#add_executable(TestTriangleMeshCuda TestTriangleMeshCuda.cpp)
#target_link_libraries(TestTriangleMeshCuda ${CMAKE_PROJECT_NAME}
#        ${GTEST_LIBRARIES} pthread)
#
#add_executable(TestPointCloudCuda TestPointCloudCuda.cpp)
#target_link_libraries(TestPointCloudCuda ${CMAKE_PROJECT_NAME}
#        ${GTEST_LIBRARIES} pthread)
#
#add_executable(TestNNCuda TestNNCuda.cpp)
#target_link_libraries(TestNNCuda ${CMAKE_PROJECT_NAME}
#        ${GTEST_LIBRARIES} pthread)