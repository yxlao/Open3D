//
// Created by wei on 10/20/18.
//

#include "ScalableTSDFVolumeCuda.cuh"
#include "UniformTSDFVolumeCuda.h"
namespace open3d {

template<size_t N>
__global__
void AllocateBlocksKernel(ScalableTSDFVolumeCudaServer<N> server,
                          ImageCudaServer<Vector1f> depth,
                          MonoPinholeCameraCuda camera,
                          TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= depth.width_ || y >= depth.height_) return;
    float d = depth.get(x, y)(0);
    if (d == 0) return;

    Vector3f X_v = server.transform_world_to_volume_ * (
        transform_camera_to_world * camera.InverseProjection(x, y, d));

    // if server.volume_to_block() not allocated
    // server.new(block idx)
}

template<size_t N>
__global__
void GetBlocksInFrustumKernel(ScalableTSDFVolumeCudaServer<N> server,
                              MonoPinholeCameraCuda camera,
                              TransformCuda transform_camera_to_world) {
    const int bucket_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bucket_idx >= server.bucket_count_) return;

    auto& hash_table = server.hash_table();

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        HashEntry<Vector3i> &entry = hash_table.entry_array().get(
            bucket_base_idx + i);
        if (entry.value_ptr != NULLPTR_CUDA) {
            // AND IN FRUSTUM
            // TODO: Maintain entry array elsewhere
            hash_table.assigned_entry_array().push_back(entry);
        }
    }

    LinkedListCudaServer<HashEntry<Vector3i>> &linked_list =
        hash_table.entry_list_array().get(bucket_idx);
    int node_ptr = linked_list.head_node_ptr();
    while (node_ptr != NULLPTR_CUDA) {
        LinkedListNodeCuda<HashEntry<Vector3i>> &linked_list_node =
            linked_list.get_node(node_ptr);
        hash_table.assigned_entry_array().push_back(linked_list_node.data);
        node_ptr = linked_list_node.next_node_ptr;
    }
}

template<size_t N>
__global__
void IntegrateKernel(ScalableTSDFVolumeCudaServer<N> server,
                     ImageCudaServer<Vector1f> depth,
                     MonoPinholeCameraCuda camera,
                     TransformCuda transform_camera_to_world) {
    const int block_idx = blockIdx.x;
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int z = threadIdx.z;
}


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
