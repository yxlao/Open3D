macro(PROFILE PROFILE_CPP_NAME)
    add_executable(${PROFILE_CPP_NAME} "${PROFILE_CPP_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${PROFILE_CPP_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${PROFILE_CPP_NAME} PROPERTIES
            FOLDER "examples/Cpp/"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples")
endmacro(PROFILE)

PROFILE(ProfileRGBDOdometry ${CMAKE_PROJECT_NAME})
PROFILE(ProfileColoredICP ${CMAKE_PROJECT_NAME})
PROFILE(ProfileFGR ${CMAKE_PROJECT_NAME})
PROFILE(ProfileIntegration ${CMAKE_PROJECT_NAME})
PROFILE(VisualizeModels ${CMAKE_PROJECT_NAME})