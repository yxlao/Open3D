
add_executable(MakeFragments MakeFragments.cpp)
target_link_libraries(MakeFragments ${CMAKE_PROJECT_NAME})

add_executable(RegisterFragments RegisterFragments.cpp)
target_link_libraries(RegisterFragments ${CMAKE_PROJECT_NAME})

add_executable(RefineRegistration RefineRegistration.cpp)
target_link_libraries(RefineRegistration ${CMAKE_PROJECT_NAME})

add_executable(IntegrateScene IntegrateScene.cpp)
target_link_libraries(IntegrateScene ${CMAKE_PROJECT_NAME})