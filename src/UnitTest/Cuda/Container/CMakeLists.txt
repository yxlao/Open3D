# Containers
add_executable(TestHashTableCuda TestHashTableCuda.cpp)
target_link_libraries(TestHashTableCuda ${CMAKE_PROJECT_NAME}
        ${3RDPARTY_LIBRARIES}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestArrayCuda TestArrayCuda.cpp)
target_link_libraries(TestArrayCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)

add_executable(TestLinkedListCuda TestLinkedListCuda.cpp)
target_link_libraries(TestLinkedListCuda ${CMAKE_PROJECT_NAME}
        ${GTEST_LIBRARIES} pthread)
