add_subdirectory(Open3D)
add_subdirectory(Tools)

# if (BUILD_CUDA)
add_subdirectory(Cuda)
# endif()

# Disable this before we fix everything
#if (BUILD_UNIT_TESTS)
#    add_subdirectory(UnitTest)
#endif ()

if (BUILD_PYTHON_MODULE)
    add_subdirectory(Python)
endif ()
