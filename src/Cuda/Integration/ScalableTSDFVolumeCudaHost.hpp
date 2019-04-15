//
// Created by wei on 11/9/18.
//

#include "ScalableTSDFVolumeCuda.h"
#include <Cuda/Container/HashTableCudaHost.hpp>
#include <cuda_runtime.h>

namespace open3d {
namespace cuda {
/**
 * Client end
 */

ScalableTSDFVolumeCuda::ScalableTSDFVolumeCuda() {
    N_ = -1;
    bucket_count_ = -1;
    value_capacity_ = -1;
}


ScalableTSDFVolumeCuda::ScalableTSDFVolumeCuda(
    int N,
    float voxel_length,
    float sdf_trunc,
    float max_range,
    const TransformCuda &transform_volume_to_world,
    int bucket_count,
    int value_capacity) {

    voxel_length_ = voxel_length;
    sdf_trunc_ = sdf_trunc;
    max_range_ = max_range;
    transform_volume_to_world_ = transform_volume_to_world;

    Create(N, bucket_count, value_capacity);
}


ScalableTSDFVolumeCuda::ScalableTSDFVolumeCuda(
    const ScalableTSDFVolumeCuda &other) {
    N_ = other.N_;
    device_ = other.device_;
    hash_table_ = other.hash_table_;
    active_subvolume_entry_array_ = other.active_subvolume_entry_array_;

    bucket_count_ = other.bucket_count_;
    value_capacity_ = other.value_capacity_;

    voxel_length_ = other.voxel_length_;
    sdf_trunc_ = other.sdf_trunc_;
    max_range_ = other.max_range_;
    transform_volume_to_world_ = other.transform_volume_to_world_;
}


ScalableTSDFVolumeCuda &ScalableTSDFVolumeCuda::operator=(
    const ScalableTSDFVolumeCuda &other) {
    if (this != &other) {
        Release();

        N_ = other.N_;
        device_ = other.device_;
        hash_table_ = other.hash_table_;
        active_subvolume_entry_array_ = other.active_subvolume_entry_array_;

        bucket_count_ = other.bucket_count_;
        value_capacity_ = other.value_capacity_;

        voxel_length_ = other.voxel_length_;
        sdf_trunc_ = other.sdf_trunc_;
        max_range_ = other.max_range_;
        transform_volume_to_world_ = other.transform_volume_to_world_;
    }

    return *this;
}


ScalableTSDFVolumeCuda::~ScalableTSDFVolumeCuda() {
    Release();
}


void ScalableTSDFVolumeCuda::Create(
    int N, int bucket_count, int value_capacity) {
    assert(bucket_count > 0 && value_capacity > 0);

    if (device_ != nullptr) {
        utility::PrintError("[ScalableTSDFVolumeCuda] Already created, "
                            "abort!\n");
        return;
    }

    N_ = N;
    bucket_count_ = bucket_count;
    value_capacity_ = value_capacity;
    device_ = std::make_shared<ScalableTSDFVolumeCudaDevice>();
    hash_table_.Create(bucket_count, value_capacity);
    active_subvolume_entry_array_.Create(value_capacity);

    /** Comparing to 512^3, we can hold (sparsely) at most (512^2) 8^3 cubes.
     *  That is 262144. **/
    const int NNN = N_ * N_ * N_;
    CheckCuda(cudaMalloc(&device_->tsdf_memory_pool_,
                         sizeof(float) * NNN * value_capacity));
    CheckCuda(cudaMalloc(&device_->weight_memory_pool_,
                         sizeof(uchar) * NNN * value_capacity));
    CheckCuda(cudaMalloc(&device_->color_memory_pool_,
                         sizeof(Vector3b) * NNN * value_capacity));

    CheckCuda(cudaMalloc(&device_->active_subvolume_indices_,
                         sizeof(int) * value_capacity));
    UpdateDevice();
    Reset();

    ScalableTSDFVolumeCudaKernelCaller::Create(*this);
}


void ScalableTSDFVolumeCuda::Reset() {
    assert(device_ != nullptr);

    const int NNN = N_ * N_ * N_;
    CheckCuda(cudaMemset(device_->tsdf_memory_pool_, 0,
                         sizeof(float) * NNN * value_capacity_));
    CheckCuda(cudaMemset(device_->weight_memory_pool_, 0,
                         sizeof(uchar) * NNN * value_capacity_));
    CheckCuda(cudaMemset(device_->color_memory_pool_, 0,
                         sizeof(Vector3b) * NNN * value_capacity_));
}


void ScalableTSDFVolumeCuda::Release() {
    if (device_ != nullptr && device_.use_count() == 1) {
        CheckCuda(cudaFree(device_->tsdf_memory_pool_));
        CheckCuda(cudaFree(device_->weight_memory_pool_));
        CheckCuda(cudaFree(device_->color_memory_pool_));
        CheckCuda(cudaFree(device_->active_subvolume_indices_));
    }

    device_ = nullptr;
    hash_table_.Release();
    active_subvolume_entry_array_.Release();
}


void ScalableTSDFVolumeCuda::UpdateDevice() {
    if (device_ != nullptr) {
        device_->N_ = N_;

        device_->hash_table_ = *hash_table_.device_;
        device_->active_subvolume_entry_array_ =
            *active_subvolume_entry_array_.device_;

        device_->bucket_count_ = bucket_count_;
        device_->value_capacity_ = value_capacity_;

        device_->voxel_length_ = voxel_length_;
        device_->inv_voxel_length_ = 1.0f / voxel_length_;
        device_->sdf_trunc_ = sdf_trunc_;
        device_->max_range_ = max_range_;

        device_->transform_volume_to_world_ =
            transform_volume_to_world_;
        device_->transform_world_to_volume_ =
            transform_volume_to_world_.Inverse();
    }
}

std::vector<Vector3i>
ScalableTSDFVolumeCuda::DownloadKeys() {
    assert(device_ != nullptr);

    auto keys = hash_table_.DownloadKeys();
    return std::move(keys);
}


std::pair<std::vector<Vector3i>,
          std::vector<ScalableTSDFVolumeCpuData>>
ScalableTSDFVolumeCuda::DownloadVolumes() {
    assert(device_ != nullptr);

    auto key_value_pairs = hash_table_.DownloadKeyValuePairs();
    std::vector<Vector3i> &keys = key_value_pairs.first;
    std::vector<UniformTSDFVolumeCudaDevice>
        &subvolumes_device = key_value_pairs.second;

    assert(keys.size() == subvolumes_device.size());

    std::vector<ScalableTSDFVolumeCpuData> subvolumes;
    subvolumes.resize(subvolumes_device.size());

    for (int i = 0; i < subvolumes.size(); ++i) {
        auto &subvolume = subvolumes[i];
        const size_t NNN = N_ * N_ * N_;
        subvolume.tsdf_.resize(NNN);
        subvolume.weight_.resize(NNN);
        subvolume.color_.resize(NNN);

        CheckCuda(cudaMemcpy(subvolume.tsdf_.data(),
                             subvolumes_device[i].tsdf_,
                             sizeof(float) * NNN,
                             cudaMemcpyDeviceToHost));
        CheckCuda(cudaMemcpy(subvolume.weight_.data(),
                             subvolumes_device[i].weight_,
                             sizeof(uchar) * NNN,
                             cudaMemcpyDeviceToHost));
        CheckCuda(cudaMemcpy(subvolume.color_.data(),
                             subvolumes_device[i].color_,
                             sizeof(Vector3b) * NNN,
                             cudaMemcpyDeviceToHost));
    }

    return std::make_pair(std::move(keys), std::move(subvolumes));
}


/** We can easily download occupied subvolumes in parallel
  * However, uploading is not guaranteed to be correct
  * due to thread conflicts **/

std::vector<int> ScalableTSDFVolumeCuda::UploadKeys(
    std::vector<Vector3i> &keys){

    std::vector<Vector3i> keys_to_attempt = keys;
    std::vector<int> value_addrs(keys.size());
    std::vector<int> index_map(keys.size());
    for (int i = 0; i < index_map.size(); ++i) {
        index_map[i] = i;
    }

    const int kTotalAttempt = 10;
    int attempt = 0;
    while (attempt++ < kTotalAttempt) {
        hash_table_.ResetLocks();
        std::vector<int> ret_value_addrs = hash_table_.New(keys_to_attempt);

        std::vector<int> new_index_map;
        std::vector<Vector3i> new_keys_to_attempt;
        for (int i = 0; i < keys_to_attempt.size(); ++i) {
            int addr = ret_value_addrs[i];
            /** Failed to allocate due to thread locks **/
            if (addr < 0) {
                new_index_map.emplace_back(index_map[i]);
                new_keys_to_attempt.emplace_back(keys_to_attempt[i]);
            } else {
                value_addrs[index_map[i]] = addr;
            }
        }

        utility::PrintInfo("%d / %d subvolume info uploaded\n",
                           keys_to_attempt.size() - new_keys_to_attempt.size(),
                           keys_to_attempt.size());

        if (new_keys_to_attempt.empty()) {
            break;
        }

        std::swap(index_map, new_index_map);
        std::swap(keys_to_attempt, new_keys_to_attempt);
    }

    if (attempt == kTotalAttempt) {
        utility::PrintWarning("Reach maximum attempts, "
                              "%d subvolumes may fail to be inserted!\n",
                              keys_to_attempt.size());
    }

    return std::move(value_addrs);
}



bool ScalableTSDFVolumeCuda::UploadVolumes(
    std::vector<Vector3i> &keys,
    std::vector<ScalableTSDFVolumeCpuData> &values) {

    auto value_addrs = UploadKeys(keys);

    const int NNN = (N_ * N_ * N_);
    bool ret = true;
    for (int i = 0; i < value_addrs.size(); ++i) {
        int addr = value_addrs[i];

        if (addr < 0) {
            ret = false;
            continue;
        }

        const int offset = NNN * addr;
        CheckCuda(cudaMemcpy(&device_->tsdf_memory_pool_[offset],
                             values[i].tsdf_.data(),
                             sizeof(float) * NNN,
                             cudaMemcpyHostToDevice));
        CheckCuda(cudaMemcpy(&device_->weight_memory_pool_[offset],
                             values[i].weight_.data(),
                             sizeof(uchar) * NNN,
                             cudaMemcpyHostToDevice));
        CheckCuda(cudaMemcpy(&device_->color_memory_pool_[offset],
                             values[i].color_.data(),
                             sizeof(Vector3b) * NNN,
                             cudaMemcpyHostToDevice));
    }
    return ret;
}


void ScalableTSDFVolumeCuda::TouchSubvolumes(
    ImageCuda<float, 1> &depth,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {
    assert(device_ != nullptr);

    ScalableTSDFVolumeCudaKernelCaller::TouchSubvolumes(
        *this, depth, camera, transform_camera_to_world);
}


void ScalableTSDFVolumeCuda::GetSubvolumesInFrustum(
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {
    assert(device_ != nullptr);

    ScalableTSDFVolumeCudaKernelCaller::GetSubvolumesInFrustum(
        *this, camera, transform_camera_to_world);
}


void ScalableTSDFVolumeCuda::GetAllSubvolumes() {
    assert(device_ != nullptr);
    ScalableTSDFVolumeCudaKernelCaller::GetAllSubvolumes(*this);
}


void ScalableTSDFVolumeCuda::IntegrateSubvolumes(
    RGBDImageCuda &rgbd,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {
    assert(device_ != nullptr);

    ScalableTSDFVolumeCudaKernelCaller::IntegrateSubvolumes(
        *this, rgbd, camera, transform_camera_to_world);
}


void ScalableTSDFVolumeCuda::ResetActiveSubvolumeIndices() {
    assert(device_ != nullptr);

    CheckCuda(cudaMemset(device_->active_subvolume_indices_, 0xff,
                         sizeof(int) * value_capacity_));
}


void ScalableTSDFVolumeCuda::Integrate(
    RGBDImageCuda &rgbd,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {
    assert(device_ != nullptr);

    hash_table_.ResetLocks();
    active_subvolume_entry_array_.set_iterator(0);
    TouchSubvolumes(rgbd.depth_, camera, transform_camera_to_world);

    ResetActiveSubvolumeIndices();
    GetSubvolumesInFrustum(camera, transform_camera_to_world);
    IntegrateSubvolumes(rgbd, camera, transform_camera_to_world);
}


void ScalableTSDFVolumeCuda::RayCasting(
    ImageCuda<float, 3> &image,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {
    assert(device_ != nullptr);

    ScalableTSDFVolumeCudaKernelCaller::RayCasting(
        *this, image, camera, transform_camera_to_world);
}

void ScalableTSDFVolumeCuda::VolumeRendering(
    ImageCuda<float, 3> &image,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {
    assert(device_ != nullptr);

    ScalableTSDFVolumeCudaKernelCaller::VolumeRendering(
        *this, image, camera, transform_camera_to_world);
}

ScalableTSDFVolumeCuda ScalableTSDFVolumeCuda::DownSample() {
    ScalableTSDFVolumeCuda volume_down(
        N_ / 2, voxel_length_ * 2, sdf_trunc_ * 2);

    auto keys = DownloadKeys();
    volume_down.UploadKeys(keys);

    GetAllSubvolumes();
    ScalableTSDFVolumeCudaKernelCaller::DownSample(*this, volume_down);

    return volume_down;
}
} // cuda
} // open3d