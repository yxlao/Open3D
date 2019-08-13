if (BUILD_AZURE_KINECT)
    # Conditionally include header files in Open3D.h
    set(BUILD_AZURE_KINECT_COMMENT "")

    if (WIN32)
        include_directories("C:/Program\ Files/Azure Kinect SDK v1.1.1/sdk/include/k4a")
        include_directories("C:/Program\ Files/Azure Kinect SDK v1.1.1/sdk/include/k4arecord")
        link_directories("C:/Program\ Files/Azure Kinect SDK v1.1.1/sdk/windows-desktop/amd64/release/bin")
    else()
        find_package(k4a QUIET)
        find_package(k4arecord QUIET)
        if (k4a_FOUND)
            message(STATUS "Enabling Azure Kinect Support: ${k4a_INCLUDE_DIRS}")
        else ()
            message(FATAL_ERROR "Kinect SDK NOT found. Please install according \
                    to https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md")
        endif ()
    endif()
else()
    set(BUILD_AZURE_KINECT_COMMENT "//")
endif()

set(BUILD_AZURE_KINECT_COMMENT ${BUILD_AZURE_KINECT_COMMENT} PARENT_SCOPE)
