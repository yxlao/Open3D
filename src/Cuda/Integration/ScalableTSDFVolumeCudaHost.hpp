//
// Created by wei on 11/9/18.
//

#include "ScalableTSDFVolumeCuda.h"
#include <Cuda/Container/HashTableCudaHost.hpp>
#include <cuda_runtime.h>

namespace open3d {
/**
 * Client end
 */
template<size_t N>
ScalableTSDFVolumeCuda<N>::ScalableTSDFVolumeCuda() {
    bucket_count_ = -1;
    value_capacity_ = -1;
}

template<size_t N>
ScalableTSDFVolumeCuda<N>::ScalableTSDFVolumeCuda(
    int bucket_count,
    int value_capacity,
    float voxel_length,
    float sdf_trunc,
    TransformCuda &transform_volume_to_world) {

    voxel_length_ = voxel_length;
    sdf_trunc_ = sdf_trunc;
    transform_volume_to_world_ = transform_volume_to_world;

    Create(bucket_count, value_capacity);
}

template<size_t N>
ScalableTSDFVolumeCuda<N>::ScalableTSDFVolumeCuda(
    const ScalableTSDFVolumeCuda<N> &other) {
    server_ = other.server();
    hash_table_ = other.hash_table();

    bucket_count_ = other.bucket_count_;
    value_capacity_ = other.value_capacity_;

    voxel_length_ = other.voxel_length_;
    sdf_trunc_ = other.sdf_trunc_;
    transform_volume_to_world_ = other.transform_volume_to_world_;
}

template<size_t N>
ScalableTSDFVolumeCuda<N> &ScalableTSDFVolumeCuda<N>::operator=(
    const ScalableTSDFVolumeCuda<N> &other) {
    if (this != &other) {
        Release();

        server_ = other.server();
        hash_table_ = other.hash_table();

        bucket_count_ = other.bucket_count_;
        value_capacity_ = other.value_capacity_;

        voxel_length_ = other.voxel_length_;
        sdf_trunc_ = other.sdf_trunc_;
        transform_volume_to_world_ = other.transform_volume_to_world_;
    }

    return *this;
}

template<size_t N>
ScalableTSDFVolumeCuda<N>::~ScalableTSDFVolumeCuda() {
    Release();
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::Create(
    int bucket_count, int value_capacity) {
    assert(bucket_count > 0 && value_capacity > 0);

    if (server_ != nullptr) {
        PrintError("[ScalableTSDFVolumeCuda] Already created, abort!\n");
        return;
    }

    bucket_count_ = bucket_count;
    value_capacity_ = value_capacity;
    server_ = std::make_shared<ScalableTSDFVolumeCudaServer<N>>();
    hash_table_.Create(bucket_count, value_capacity);
    active_subvolume_entry_array_.Create(value_capacity);

    /** Comparing to 512^3, we can hold (sparsely) at most (512^2) 8^3 cubes.
     *  That is 262144. **/
    const int NNN = N * N * N;
    CheckCuda(cudaMalloc(&server_->tsdf_memory_pool_,
                         sizeof(float) * NNN * value_capacity));
    CheckCuda(cudaMalloc(&server_->weight_memory_pool_,
                         sizeof(uchar) * NNN * value_capacity));
    CheckCuda(cudaMalloc(&server_->color_memory_pool_,
                         sizeof(Vector3b) * NNN * value_capacity));

    CheckCuda(cudaMalloc(&server_->active_subvolume_indices_,
                         sizeof(int) * value_capacity));
    UpdateServer();
    Reset();

    ScalableTSDFVolumeCudaKernelCaller<N>::
    CreateScalableTSDFVolumesKernelCaller(
        *server_, value_capacity_);
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::Reset() {
    assert(server_ != nullptr);

    const int NNN = N * N * N;
    CheckCuda(cudaMemset(server_->tsdf_memory_pool_, 0,
                         sizeof(float) * NNN * value_capacity_));
    CheckCuda(cudaMemset(server_->weight_memory_pool_, 0,
                         sizeof(uchar) * NNN * value_capacity_));
    CheckCuda(cudaMemset(server_->color_memory_pool_, 0,
                         sizeof(Vector3b) * NNN * value_capacity_));
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        CheckCuda(cudaFree(server_->tsdf_memory_pool_));
        CheckCuda(cudaFree(server_->weight_memory_pool_));
        CheckCuda(cudaFree(server_->color_memory_pool_));
        CheckCuda(cudaFree(server_->active_subvolume_indices_));
    }

    server_ = nullptr;
    hash_table_.Release();
    active_subvolume_entry_array_.Release();
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::UpdateServer() {
    if (server_ != nullptr) {
        server_->hash_table_ = *hash_table_.server();
        server_->active_subvolume_entry_array_ =
            *active_subvolume_entry_array_.server();

        server_->bucket_count_ = bucket_count_;
        server_->value_capacity_ = value_capacity_;

        server_->voxel_length_ = voxel_length_;
        server_->inv_voxel_length_ = 1.0f / voxel_length_;
        server_->sdf_trunc_ = sdf_trunc_;
        server_->transform_volume_to_world_ =
            transform_volume_to_world_;
        server_->transform_world_to_volume_ =
            transform_volume_to_world_.Inverse();
    }
}

template<size_t N>
std::pair<std::vector<Vector3i>,
std::vector<std::tuple<std::vector<float>,
std::vector<uchar>,
std::vector<Vector3b>>>>
ScalableTSDFVolumeCuda<N>::DownloadVolumes() {
    assert(server_ != nullptr);

    auto hash_table = hash_table_.Download();
    std::vector<Vector3i> &keys = std::get<0>(hash_table);
    std::vector<UniformTSDFVolumeCudaServer<N>>
        &volume_servers = std::get<1>(hash_table);

    assert(keys.size() == volume_servers.size());

    std::vector<std::tuple<std::vector<float>,
        std::vector<uchar>,
        std::vector<Vector3b>>> volumes;
    volumes.resize(volume_servers.size());

    for (int i = 0; i < volumes.size(); ++i) {
        std::vector<float> tsdf;
        std::vector<uchar> weight;
        std::vector<Vector3b> color;

        const size_t NNN = N * N * N;
        tsdf.resize(NNN);
        weight.resize(NNN);
        color.resize(NNN);

        CheckCuda(cudaMemcpy(tsdf.data(), volume_servers[i].tsdf_,
                             sizeof(float) * NNN,
                             cudaMemcpyDeviceToHost));
        CheckCuda(cudaMemcpy(weight.data(), volume_servers[i].weight_,
                             sizeof(uchar) * NNN,
                             cudaMemcpyDeviceToHost));
        CheckCuda(cudaMemcpy(color.data(), volume_servers[i].color_,
                             sizeof(Vector3b) * NNN,
                             cudaMemcpyDeviceToHost));

        volumes[i] = std::make_tuple(
            std::move(tsdf), std::move(weight), std::move(color));
    }

    return std::make_pair(std::move(keys), std::move(volumes));
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::TouchSubvolumes(
    ImageCuda<Vector1f> &depth,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {
    assert(server_ != nullptr);

    ScalableTSDFVolumeCudaKernelCaller<N>::
    TouchSubvolumesKernelCaller(
        *server_, *depth.server(), camera, transform_camera_to_world);
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::GetSubvolumesInFrustum(
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {
    assert(server_ != nullptr);

    ScalableTSDFVolumeCudaKernelCaller<N>::
    GetSubvolumesInFrustumKernelCaller(
        *server_, camera, transform_camera_to_world, bucket_count_);
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::GetAllSubvolumes() {
    assert(server_ != nullptr);
    ScalableTSDFVolumeCudaKernelCaller<N>::
    GetAllSubvolumesKernelCaller(*server_, bucket_count_);
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::IntegrateSubvolumes(
    RGBDImageCuda &rgbd,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {
    assert(server_ != nullptr);

    const int active_subvolumes = active_subvolume_entry_array_.size();
    ScalableTSDFVolumeCudaKernelCaller<N>::
    IntegrateSubvolumesKernelCaller(
        *server_, *rgbd.server(), camera, transform_camera_to_world,
        active_subvolumes);
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::ResetActiveSubvolumeIndices() {
    assert(server_ != nullptr);

    CheckCuda(cudaMemset(server_->active_subvolume_indices_, 0xff,
                         sizeof(int) * value_capacity_));
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::Integrate(
    RGBDImageCuda &rgbd,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {
    assert(server_ != nullptr);

    hash_table_.ResetLocks();
    active_subvolume_entry_array_.set_size(0);
    TouchSubvolumes(rgbd.depthf(), camera, transform_camera_to_world);

    ResetActiveSubvolumeIndices();
    GetSubvolumesInFrustum(camera, transform_camera_to_world);
    IntegrateSubvolumes(rgbd, camera, transform_camera_to_world);
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::RayCasting(
    ImageCuda<Vector3f> &image,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {
    assert(server_ != nullptr);

    ScalableTSDFVolumeCudaKernelCaller<N>::
    RayCastingKernelCaller(
        *server_, *image.server(), camera, transform_camera_to_world);
}
}