
add_executable(MakeFragments MakeFragments.cpp)
target_link_libraries(MakeFragments ${CMAKE_PROJECT_NAME})

add_executable(RegisterFragments RegisterFragments.cpp)
target_link_libraries(RegisterFragments ${CMAKE_PROJECT_NAME})

add_executable(RegisterFragmentsWithReloc RegisterFragmentsWithReloc.cpp)
target_link_libraries(RegisterFragmentsWithReloc ${CMAKE_PROJECT_NAME})

add_executable(RefineRegistration RefineRegistration.cpp)
target_link_libraries(RefineRegistration ${CMAKE_PROJECT_NAME})

add_executable(IntegrateScene IntegrateScene.cpp)
target_link_libraries(IntegrateScene ${CMAKE_PROJECT_NAME})

add_executable(ViewFragments ViewFragments.cpp)
target_link_libraries(ViewFragments ${CMAKE_PROJECT_NAME})

add_executable(ViewPoses ViewPoseGraph.cpp)
target_link_libraries(ViewPoses ${CMAKE_PROJECT_NAME})

add_executable(RunSystem RunSystem.cpp)
target_link_libraries(RunSystem ${CMAKE_PROJECT_NAME})