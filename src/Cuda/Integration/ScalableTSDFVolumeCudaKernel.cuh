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

    const Vector2i p = Vector2i(x, y);
    server.TouchSubvolume(p, depth, camera, transform_camera_to_world);
}

template<size_t N>
__global__
void IntegrateSubvolumesKernel(ScalableTSDFVolumeCudaServer<N> server,
                               RGBDImageCudaServer rgbd,
                               MonoPinholeCameraCuda camera,
                               TransformCuda transform_camera_to_world) {

    const size_t entry_idx = blockIdx.x;
    const Vector3i Xlocal = Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(entry_idx < server.active_subvolume_entry_array().size()
        && Xlocal(0) < N && Xlocal(1) < N && Xlocal(2) < N);
#endif

    HashEntry<Vector3i>
        &entry = server.active_subvolume_entry_array().at(entry_idx);
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(entry.internal_addr >= 0);
#endif
    server.Integrate(Xlocal, entry, rgbd, camera, transform_camera_to_world);
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

    Vector2i p = Vector2i(x, y);
    Vector3f n = server.RayCasting(p, camera, transform_camera_to_world);
    normal.get(x, y) = (n == Vector3f::Zeros()) ? n : 0.5f * n + Vector3f(0.5f);
}

template<size_t N>
__global__
void CreateScalableTSDFVolumesKernel(ScalableTSDFVolumeCudaServer<N> server) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= server.value_capacity_) return;

    const size_t offset = (N * N * N) * index;
    UniformTSDFVolumeCudaServer<N> &subvolume = server.hash_table()
        .memory_heap_value().value_at(index);

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
    for (size_t i = 0; i < BUCKET_SIZE; ++i) {
        HashEntry<Vector3i> &entry = hash_table.entry_array().at(
            bucket_base_idx + i);
        if (entry.internal_addr != NULLPTR_CUDA) {
            Vector3f X = server.voxelf_local_to_global(Vector3f(0), entry.key);
            if (camera.IsInFrustum(
                transform_camera_to_world.Inverse()
                    * server.voxelf_to_world(X))) {
                server.ActivateSubvolume(entry);
            }
        }
    }

    LinkedListCudaServer<HashEntry<Vector3i>> &linked_list =
        hash_table.entry_list_array().at(bucket_idx);
    int node_ptr = linked_list.head_node_ptr();
    while (node_ptr != NULLPTR_CUDA) {
        LinkedListNodeCuda<HashEntry<Vector3i>> &linked_list_node =
            linked_list.get_node(node_ptr);

        HashEntry<Vector3i> &entry = linked_list_node.data;
        Vector3f X = server.voxelf_local_to_global(Vector3f(0), entry.key);
        if (camera.IsInFrustum(
            transform_camera_to_world.Inverse() * server.voxelf_to_world(X))) {
            server.ActivateSubvolume(entry);
        }

        node_ptr = linked_list_node.next_node_ptr;
    }
}

template<size_t N>
__global__
void GetAllSubvolumesKernel(ScalableTSDFVolumeCudaServer<N> server) {
    const int bucket_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bucket_idx >= server.bucket_count_) return;

    auto &hash_table = server.hash_table();

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
#pragma unroll 1
    for (size_t i = 0; i < BUCKET_SIZE; ++i) {
        HashEntry<Vector3i> &entry = hash_table.entry_array().at(
            bucket_base_idx + i);
        if (entry.internal_addr != NULLPTR_CUDA) {
            server.ActivateSubvolume(entry);
        }
    }

    LinkedListCudaServer<HashEntry<Vector3i>> &linked_list =
        hash_table.entry_list_array().at(bucket_idx);
    int node_ptr = linked_list.head_node_ptr();
    while (node_ptr != NULLPTR_CUDA) {
        LinkedListNodeCuda<HashEntry<Vector3i>> &linked_list_node =
            linked_list.get_node(node_ptr);
        server.ActivateSubvolume(linked_list_node.data);
        node_ptr = linked_list_node.next_node_ptr;
    }
}
}
