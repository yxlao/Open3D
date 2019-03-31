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

find_package(CUDA 8.0 REQUIRED)
find_package(OpenCV REQUIRED)
list(APPEND 3RDPARTY_INCLUDE_DIRS ${CUDA_TOOLKIT_INCLUDE}
        ${OpenCV_INCLUDE_DIRS})
list(APPEND 3RDPARTY_LIBRARIES
        ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY}
        ${OpenCV_LIBS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${3RDPARTY_INCLUDE_DIRS})
link_directories(${3RDPARTY_LIBRARY_DIRS})

add_subdirectory(Demo)
add_subdirectory(ReconstructionSystem)
add_subdirectory(Experiment)
