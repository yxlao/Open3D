//
// Created by wei on 11/9/18.
//

#include "UniformTSDFVolumeCuda.h"

#include <Cuda/Common/UtilsCuda.h>

#include <Core/Core.h>

namespace open3d {
namespace cuda {

/**
 * Client end
 */
template<size_t N>
UniformTSDFVolumeCuda<N>::UniformTSDFVolumeCuda() {}

template<size_t N>
UniformTSDFVolumeCuda<N>::UniformTSDFVolumeCuda(
    float voxel_length, float sdf_trunc,
    TransformCuda &volume_to_world) {

    voxel_length_ = voxel_length;
    sdf_trunc_ = sdf_trunc;
    transform_volume_to_world_ = volume_to_world;

    Create();
}

template<size_t N>
UniformTSDFVolumeCuda<N>::UniformTSDFVolumeCuda(
    const UniformTSDFVolumeCuda<N> &other) {

    device_ = other.device_;
    voxel_length_ = other.voxel_length_;
    sdf_trunc_ = other.sdf_trunc_;
    transform_volume_to_world_ = other.transform_volume_to_world_;
}

template<size_t N>
UniformTSDFVolumeCuda<N> &UniformTSDFVolumeCuda<N>::operator=(
    const UniformTSDFVolumeCuda<N> &other) {
    if (this != &other) {
        device_ = other.device_;
        voxel_length_ = other.voxel_length_;
        sdf_trunc_ = other.sdf_trunc_;
        transform_volume_to_world_ = other.transform_volume_to_world_;
    }
    return *this;
}

template<size_t N>
UniformTSDFVolumeCuda<N>::~UniformTSDFVolumeCuda() {
    Release();
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::Create() {
    if (device_ != nullptr) {
        PrintError("[UniformTSDFVolumeCuda] Already created, abort!\n");
        return;
    }

    device_ = std::make_shared<UniformTSDFVolumeCudaDevice<N>>();
    const size_t NNN = N * N * N;
    CheckCuda(cudaMalloc(&(device_->tsdf_), sizeof(float) * NNN));
    CheckCuda(cudaMalloc(&(device_->weight_), sizeof(uchar) * NNN));
    CheckCuda(cudaMalloc(&(device_->color_), sizeof(Vector3b) * NNN));

    UpdateDevice();
    Reset();
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::Release() {
    if (device_ != nullptr && device_.use_count() == 1) {
        CheckCuda(cudaFree(device_->tsdf_));
        CheckCuda(cudaFree(device_->weight_));
        CheckCuda(cudaFree(device_->color_));
    }

    device_ = nullptr;
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::UpdateDevice() {
    if (device_ != nullptr) {
        device_->voxel_length_ = voxel_length_;
        device_->inv_voxel_length_ = 1.0f / voxel_length_;

        device_->sdf_trunc_ = sdf_trunc_;
        device_->transform_volume_to_world_ = transform_volume_to_world_;
        device_->transform_world_to_volume_ =
            transform_volume_to_world_.Inverse();
    }
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::Reset() {
    if (device_ != nullptr) {
        const size_t NNN = N * N * N;
        CheckCuda(cudaMemset(device_->tsdf_, 0, sizeof(float) * NNN));
        CheckCuda(cudaMemset(device_->weight_, 0, sizeof(uchar) * NNN));
        CheckCuda(cudaMemset(device_->color_, 0, sizeof(Vector3b) * NNN));
    }
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::UploadVolume(std::vector<float> &tsdf,
                                            std::vector<uchar> &weight,
                                            std::vector<Vector3b> &color) {
    assert(device_ != nullptr);

    const size_t NNN = N * N * N;
    assert(tsdf.size() == NNN);
    assert(weight.size() == NNN);
    assert(color.size() == NNN);

    CheckCuda(cudaMemcpy(device_->tsdf_, tsdf.data(),
                         sizeof(float) * NNN,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(device_->weight_, weight.data(),
                         sizeof(uchar) * NNN,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(device_->color_, color.data(),
                         sizeof(Vector3b) * NNN,
                         cudaMemcpyHostToDevice));
}

template<size_t N>
std::tuple<std::vector<float>, std::vector<uchar>, std::vector<Vector3b>>
UniformTSDFVolumeCuda<N>::DownloadVolume() {
    assert(device_ != nullptr);

    std::vector<float> tsdf;
    std::vector<uchar> weight;
    std::vector<Vector3b> color;

    if (device_ == nullptr) {
        PrintError("Server not available!\n");
        return std::make_tuple(tsdf, weight, color);
    }

    const size_t NNN = N * N * N;
    tsdf.resize(NNN);
    weight.resize(NNN);
    color.resize(NNN);

    CheckCuda(cudaMemcpy(tsdf.data(), device_->tsdf_,
                         sizeof(float) * NNN,
                         cudaMemcpyDeviceToHost));
    CheckCuda(cudaMemcpy(weight.data(), device_->weight_,
                         sizeof(uchar) * NNN,
                         cudaMemcpyDeviceToHost));
    CheckCuda(cudaMemcpy(color.data(), device_->color_,
                         sizeof(Vector3b) * NNN,
                         cudaMemcpyDeviceToHost));

    return std::make_tuple(
        std::move(tsdf), std::move(weight), std::move(color));
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::Integrate(RGBDImageCuda &rgbd,
                                         PinholeCameraIntrinsicCuda &camera,
                                         TransformCuda &transform_camera_to_world) {
    assert(device_ != nullptr);
    UniformTSDFVolumeCudaKernelCaller<N>::Integrate(
        *this, rgbd, camera, transform_camera_to_world);
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::RayCasting(ImageCuda<Vector3f> &image,
                                          PinholeCameraIntrinsicCuda &camera,
                                          TransformCuda &transform_camera_to_world) {
    assert(device_ != nullptr);
    UniformTSDFVolumeCudaKernelCaller<N>::RayCasting(
        *this, image, camera, transform_camera_to_world);
}
} // cuda
} // open3d