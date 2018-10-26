//
// Created by wei on 10/20/18.
//

#include "ScalableTSDFVolumeCuda.cuh"
#include "UniformTSDFVolumeCuda.h"
namespace open3d {

template<size_t N>
__global__
void TouchSubvolumesKernel(ScalableTSDFVolumeCudaServer<N> server,
                           ImageCudaServer<Vector1f> depth,
                           MonoPinholeCameraCuda camera,
                           TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= depth.width_ || y >= depth.height_) return;

    server.TouchSubvolume(x, y, depth, camera, transform_camera_to_world);
}

template<size_t N>
__global__
void IntegrateSubvolumesKernel(ScalableTSDFVolumeCudaServer<N> server,
                     ImageCudaServer<Vector1f> depth,
                     MonoPinholeCameraCuda camera,
                     TransformCuda transform_camera_to_world) {

    const size_t entry_idx = blockIdx.x;
    const int xlocal = threadIdx.x;
    const int ylocal = threadIdx.y;
    const int zlocal = threadIdx.z;

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert (entry_idx < server.active_subvolume_entry_array().size()
        && xlocal < N && ylocal < N && zlocal < N);
#endif

    HashEntry<Vector3i> &entry = server.active_subvolume_entry_array().get(
        entry_idx);
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(entry.internal_addr >= 0);
#endif
    server.Integrate(xlocal, ylocal, zlocal, entry,
                     depth, camera, transform_camera_to_world);
}

template<size_t N>
__global__
void RayCastingKernel(ScalableTSDFVolumeCudaServer<N> server,
                      ImageCudaServer<Vector3f> normal,
                      MonoPinholeCameraCuda camera,
                      TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= normal.width_ || y >= normal.height_) return;

    Vector3f n = server.RayCasting(x, y, camera, transform_camera_to_world);
    normal.get(x, y) = (n == Vector3f::Zeros()) ? n : 0.5f * n + Vector3f(0.5f);
}

template<size_t N>
__global__
void CreateScalableTSDFVolumesKernel(ScalableTSDFVolumeCudaServer<N> server) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= server.value_capacity_) return;

    const size_t offset = (N * N * N) * index;
    UniformTSDFVolumeCudaServer<N> &subvolume = server.hash_table()
        .memory_heap_value().get_value(index);

    /** Assign memory **/
    subvolume.Create(&server.tsdf_memory_pool()[offset],
                     &server.weight_memory_pool()[offset],
                     &server.color_memory_pool()[offset]);

    /** Assign property **/
    subvolume.voxel_length_ = server.voxel_length_;
    subvolume.inv_voxel_length_ = server.inv_voxel_length_;
    subvolume.sdf_trunc_ = server.sdf_trunc_;
    subvolume.transform_volume_to_world_ = server.transform_volume_to_world_;
    subvolume.transform_world_to_volume_ = server.transform_world_to_volume_;
}


template<size_t N>
__global__
void GetSubvolumesInFrustumKernel(ScalableTSDFVolumeCudaServer<N> server,
                                  MonoPinholeCameraCuda camera,
                                  TransformCuda transform_camera_to_world) {
    const int bucket_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bucket_idx >= server.bucket_count_) return;

    auto &hash_table = server.hash_table();

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
#pragma unroll 1
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        HashEntry<Vector3i> &entry = hash_table.entry_array().get(
            bucket_base_idx + i);
        if (entry.internal_addr != NULLPTR_CUDA) {
            Vector3f X = server.voxelf_local_to_global(Vector3f(0), entry.key);
            if (camera.IsInFrustum(server.voxelf_to_world(X))) {
                server.ActivateSubvolume(entry);
            }
        }
    }

    LinkedListCudaServer<HashEntry<Vector3i>> &linked_list =
        hash_table.entry_list_array().get(bucket_idx);
    int node_ptr = linked_list.head_node_ptr();
    while (node_ptr != NULLPTR_CUDA) {
        LinkedListNodeCuda<HashEntry<Vector3i>> &linked_list_node =
            linked_list.get_node(node_ptr);

        HashEntry<Vector3i> &entry = linked_list_node.data;
        Vector3f X = server.voxelf_local_to_global(Vector3f(0), entry.key);
        if (camera.IsInFrustum(server.voxelf_to_world(X))) {
            server.ActivateSubvolume(entry);
        }

        node_ptr = linked_list_node.next_node_ptr;
    }
}
}
