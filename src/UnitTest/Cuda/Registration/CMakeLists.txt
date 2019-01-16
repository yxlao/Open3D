add_executable(TestCorrespondenceSetCuda TestCorrespondenceSetCuda.cpp)
target_link_libraries(TestCorrespondenceSetCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestColoredICPCuda TestColoredICPCuda.cpp)
target_link_libraries(TestColoredICPCuda ${GTEST_LIBRARIES} pthread)