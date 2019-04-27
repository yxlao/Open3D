add_subdirectory(Open3D)
add_subdirectory(Tools)

# > Enable this by default in this branch
if (BUILD_CUDA_MODULE)
    add_subdirectory(Cuda)
    add_subdirectory(InverseRendering)
endif()

# > Disable this before we fix everything
if (BUILD_UNIT_TESTS)
    add_subdirectory(UnitTest)
endif ()

if (BUILD_PYTHON_MODULE)
    add_subdirectory(Python)
endif ()
