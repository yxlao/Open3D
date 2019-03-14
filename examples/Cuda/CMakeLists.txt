macro(EXAMPLE_CUDA EXAMPLE_CUDA_NAME)
    add_executable(${EXAMPLE_CUDA_NAME} "${EXAMPLE_CUDA_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${EXAMPLE_CUDA_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${EXAMPLE_CUDA_NAME} PROPERTIES
            FOLDER "examples/Cuda/"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples")
endmacro(EXAMPLE_CUDA)

add_subdirectory(Demo)
add_subdirectory(Profile)
add_subdirectory(ReconstructionSystem)

