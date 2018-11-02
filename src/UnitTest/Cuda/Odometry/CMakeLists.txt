# Odometry
add_executable(TestReduction2DCuda TestReduction2DCuda.cpp)
target_link_libraries(TestReduction2DCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestRGBDOdometryCuda TestRGBDOdometryCuda.cpp)
target_link_libraries(TestRGBDOdometryCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)
