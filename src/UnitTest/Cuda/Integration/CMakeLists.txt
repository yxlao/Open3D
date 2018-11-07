# Integration
add_executable(TestUniformTSDFVolumeCuda TestUniformTSDFVolumeCuda.cpp)
target_link_libraries(TestUniformTSDFVolumeCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestUniformMeshVolumeCuda TestUniformMeshVolumeCuda.cpp)
target_link_libraries(TestUniformMeshVolumeCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestScalableTSDFVolumeCuda TestScalableTSDFVolumeCuda.cpp)
target_link_libraries(TestScalableTSDFVolumeCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestScalableMeshVolumeCuda TestScalableMeshVolumeCuda.cpp)
target_link_libraries(TestScalableMeshVolumeCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestScalableSystem TestScalableSystem.cpp)
target_link_libraries(TestScalableSystem ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestUniformSystem TestUniformSystem.cpp)
target_link_libraries(TestUniformSystem ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)