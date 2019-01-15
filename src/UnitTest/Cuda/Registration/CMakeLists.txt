add_executable(TestCorrespondenceSetCuda TestCorrespondenceSetCuda.cpp)
target_link_libraries(TestCorrespondenceSetCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)