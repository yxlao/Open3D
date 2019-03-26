# CPP examples
if (BUILD_CPP_EXAMPLES)
    add_subdirectory(Cpp)
endif ()

# Cuda module -- Enable it here
add_subdirectory(Cuda)

# Python examples
if (BUILD_PYTHON_MODULE)
    add_subdirectory(Python)
endif ()
