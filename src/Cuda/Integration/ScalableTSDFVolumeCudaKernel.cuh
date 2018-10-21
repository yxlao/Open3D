//
// Created by wei on 10/20/18.
//

#include "ScalableTSDFVolumeCuda.cuh"
#include "UniformTSDFVolumeCuda.h"
namespace open3d {

template<size_t N>
__global__
void CreateScalableTSDFVolumesKernel(ScalableTSDFVolumeCudaServer<N> server) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= server.value_capacity_) return;

    const size_t offset = (N * N * N) * index;
    UniformTSDFVolumeCudaServer<N> &uniform_volume = server.hash_table()
        .memory_heap_value().get_value(index);

    /** Assign memory **/
    uniform_volume.Create(server.tsdf_memory_pool() + offset,
                          server.weight_memory_pool() + offset,
                          server.color_memory_pool() + offset);

    /** Assign property **/
    uniform_volume.voxel_length_ = server.voxel_length_;
    uniform_volume.inv_voxel_length_ = server.inv_voxel_length_;
    uniform_volume.sdf_trunc_ = server.sdf_trunc_;
    uniform_volume.transform_volume_to_world_ = server.transform_volume_to_world_;
    uniform_volume.transform_world_to_volume_ = server.transform_world_to_volume_;
}
}
