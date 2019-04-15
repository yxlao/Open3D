# CPP examples
if (BUILD_CPP_EXAMPLES)
    add_subdirectory(Cpp)
endif ()

# Cuda module -- Enable it here
if (BUILD_CUDA_EXAMPLES)
    add_subdirectory(Cuda)
endif()

# Python examples
if (BUILD_PYTHON_MODULE)
    add_subdirectory(Python)
endif ()

# Experiment
add_subdirectory(Material)