find_package(CUDA 8.0 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)

file(GLOB CUDA_ALL_HEADER_FILES '*.h')
file(GLOB CUDA_ALL_SOURCE_FILES '*.cpp')
file(GLOB CUDA_ALL_CUDA_HEADER_FILES '*.cuh')
file(GLOB CUDA_ALL_CUDA_SOURCE_FILES '*.cu')

file(GLOB CUDA_COMMON_HEADER_FILES "Common/*.h")
source_group("Header Files\\Common" FILES ${CUDA_COMMON_HEADER_FILES})
file(GLOB CUDA_COMMON_SOURCE_FILES "Common/*.cpp")
source_group("Source Files\\Common" FILES ${CUDA_COMMON_SOURCE_FILES})
file(GLOB CUDA_COMMON_CUDA_HEADER_FILES "Common/*.cuh")
source_group("Header Files\\Common" FILES ${CUDA_COMMON_CUDA_HEADER_FILES})
file(GLOB CUDA_COMMON_CUDA_SOURCE_FILES "Common/*.cu")
source_group("Source Files\\Common" FILES ${CUDA_COMMON_CUDA_SOURCE_FILES})

file(GLOB CUDA_CONTAINER_HEADER_FILES "Container/*.h")
source_group("Header Files\\Container" FILES ${CUDA_CONTAINER_HEADER_FILES})
file(GLOB CUDA_CONTAINER_SOURCE_FILES "Container/*.cpp")
source_group("Source Files\\Container" FILES ${CUDA_CONTAINER_SOURCE_FILES})
file(GLOB CUDA_CONTAINER_CUDA_HEADER_FILES "Container/*.cuh")
source_group("Header Files\\Container" FILES ${CUDA_CONTAINER_CUDA_HEADER_FILES})
file(GLOB CUDA_CONTAINER_CUDA_SOURCE_FILES "Container/*.cu")
source_group("Source Files\\Container" FILES ${CUDA_CONTAINER_CUDA_SOURCE_FILES})

file(GLOB CUDA_GEOMETRY_HEADER_FILES "Geometry/*.h")
source_group("Header Files\\Geometry" FILES ${CUDA_GEOMETRY_HEADER_FILES})
file(GLOB CUDA_GEOMETRY_SOURCE_FILES "Geometry/*.cpp")
source_group("Source Files\\Geometry" FILES ${CUDA_GEOMETRY_SOURCE_FILES})
file(GLOB CUDA_GEOMETRY_CUDA_HEADER_FILES "Geometry/*.cuh")
source_group("Header Files\\Geometry" FILES ${CUDA_GEOMETRY_CUDA_HEADER_FILES})
file(GLOB CUDA_GEOMETRY_CUDA_SOURCE_FILES "Geometry/*.cu")
source_group("Source Files\\Geometry" FILES ${CUDA_GEOMETRY_CUDA_SOURCE_FILES})

file(GLOB CUDA_ODOMETRY_HEADER_FILES "Odometry/*.h")
source_group("Header Files\\Odometry" FILES ${CUDA_ODOMETRY_HEADER_FILES})
file(GLOB CUDA_ODOMETRY_SOURCE_FILES "Odometry/*.cpp")
source_group("Source Files\\Odometry" FILES ${CUDA_ODOMETRY_SOURCE_FILES})
file(GLOB CUDA_ODOMETRY_CUDA_HEADER_FILES "Odometry/*.cuh")
source_group("Header Files\\Odometry" FILES ${CUDA_ODOMETRY_CUDA_HEADER_FILES})
file(GLOB CUDA_ODOMETRY_CUDA_SOURCE_FILES "Odometry/*.cu")
source_group("Source Files\\Odometry" FILES ${CUDA_ODOMETRY_CUDA_SOURCE_FILES})

set(CUDA_NVCC_FLAGS
		-gencode=arch=compute_50,code=sm_60
		-gencode=arch=compute_50,code=sm_50 # simple test for multiple targets
		--expt-relaxed-constexpr)

cuda_add_library(Cuda ${Open3D_LIB_TYPE}
		${CUDA_CONTAINER_HEADER_FILES}
		${CUDA_CONTAINER_CUDA_HEADER_FILES}
		${CUDA_CONTAINER_CUDA_SOURCE_FILES}

		${CUDA_GEOMETRY_HEADER_FILES}
		${CUDA_GEOMETRY_CUDA_HEADER_FILES}
		${CUDA_GEOMETRY_CUDA_SOURCE_FILES}

		${CUDA_ODOMETRY_HEADER_FILES}
		${CUDA_ODOMETRY_CUDA_HEADER_FILES}
		${CUDA_ODOMETRY_CUDA_SOURCE_FILES}

		${CUDA_COMMON_CUDA_HEADER_FILES}
		${CUDA_COMMON_CUDA_SOURCE_FILES})
target_link_libraries(Cuda ${OpenCV_LIBS})

install(TARGETS
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
		)

file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)