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

# Registration
EXAMPLE_CUDA(ColoredICPCudaForRGBD ${CMAKE_PROJECT_NAME})
EXAMPLE_CUDA(ColoredICPCudaForFragment ${CMAKE_PROJECT_NAME})
EXAMPLE_CUDA(ICPCuda ${CMAKE_PROJECT_NAME})
EXAMPLE_CUDA(FastGlobalRegistrationCuda ${CMAKE_PROJECT_NAME})
EXAMPLE_CUDA(FPFHFeatureCuda ${CMAKE_PROJECT_NAME})

# Fusion
EXAMPLE_CUDA(ScalableFusionSystemCuda ${CMAKE_PROJECT_NAME})
EXAMPLE_CUDA(UniformFusionSystemCuda ${CMAKE_PROJECT_NAME})

# Odometry
EXAMPLE_CUDA(RGBDOdometryCuda ${CMAKE_PROJECT_NAME})
EXAMPLE_CUDA(RGBDOdometryCudaVisualizer ${CMAKE_PROJECT_NAME})

add_subdirectory(Profile)
add_subdirectory(ReconstructionSystem)