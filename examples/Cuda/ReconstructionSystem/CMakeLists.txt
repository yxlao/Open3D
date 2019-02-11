add_executable(ViewFragments ViewFragments.cpp)
target_link_libraries(ViewFragments ${CMAKE_PROJECT_NAME})

add_executable(ViewPoseGraph ViewPoseGraph.cpp)
target_link_libraries(ViewPoseGraph ${CMAKE_PROJECT_NAME})

add_executable(RunSystem RunSystem.cpp)
target_link_libraries(RunSystem ${CMAKE_PROJECT_NAME})

add_executable(CVEstimation ORBPoseEstimation.h)
target_link_libraries(CVEstimation ${CMAKE_PROJECT_NAME})

add_executable(DebugICPOdometry DebugICPOdometry.cpp)
target_link_libraries(DebugICPOdometry ${CMAKE_PROJECT_NAME})

add_executable(RegisterFragmentsWithReloc RegisterFragmentsWithOdometry.cpp)
target_link_libraries(RegisterFragmentsWithReloc ${CMAKE_PROJECT_NAME})
