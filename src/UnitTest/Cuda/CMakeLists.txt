macro(UNIT_TEST_CUDA UNIT_TEST_CUDA_NAME)
    add_executable(${UNIT_TEST_CUDA_NAME} "${UNIT_TEST_CUDA_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${UNIT_TEST_CUDA_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${UNIT_TEST_CUDA_NAME} PROPERTIES
            FOLDER "examples/Cuda/"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples")
endmacro(UNIT_TEST_CUDA)


add_subdirectory(Container)
add_subdirectory(Geometry)
add_subdirectory(Integration)
add_subdirectory(Odometry)
add_subdirectory(Registration)