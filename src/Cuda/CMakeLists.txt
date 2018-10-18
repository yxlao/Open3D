# build
file(GLOB_RECURSE CUDA_ALL_HEADER_FILES "*.h")
file(GLOB_RECURSE CUDA_ALL_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE CUDA_ALL_CUDA_HEADER_FILES "*.cuh")
file(GLOB_RECURSE CUDA_ALL_CUDA_SOURCE_FILES "*.cu")

# (testing) flags for cuda code generation
set(CUDA_NVCC_FLAGS
        -gencode=arch=compute_50,code=sm_60
        -gencode=arch=compute_50,code=sm_50
        --expt-relaxed-constexpr)

# create (should be static) library
cuda_add_library(Cuda
        ${CUDA_ALL_HEADER_FILES} ${CUDA_ALL_SOURCE_FILES}
        ${CUDA_ALL_CUDA_HEADER_FILES} ${CUDA_ALL_CUDA_SOURCE_FILES}
        $<TARGET_OBJECTS:Core>)
target_link_libraries(Cuda ${3RD_PARTY_LIBRARIES})

# create subgroups
ADD_SOURCE_GROUP(Common)
ADD_SOURCE_GROUP(Container)
ADD_SOURCE_GROUP(Geometry)
ADD_SOURCE_GROUP(Integration)
ADD_SOURCE_GROUP(Odometry)

# and also for cuda
ADD_CUDA_SOURCE_GROUP(Common)
ADD_CUDA_SOURCE_GROUP(Container)
ADD_CUDA_SOURCE_GROUP(Geometry)
ADD_CUDA_SOURCE_GROUP(Integration)
ADD_CUDA_SOURCE_GROUP(Odometry)

Directories(${CMAKE_CURRENT_SOURCE_DIR} CUDA_INCLUDE_DIRS)
set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} PARENT_SCOPE)

target_include_directories(Cuda PUBLIC ${CUDA_INCLUDE_DIRS})

#install headers
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}"
        PATTERN "*.cpp" EXCLUDE
        PATTERN "*.cuh" EXCLUDE
        PATTERN "*.cu" EXCLUDE
        PATTERN "*.txt" EXCLUDE)