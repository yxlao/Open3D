//
// Created by wei on 11/9/18.
//

#pragma once

#include "ScalableTSDFVolumeProcessorCuda.h"
#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>

#include <cuda_runtime.h>

namespace open3d {
namespace cuda {
/**
 * Client end
 */
ScalableTSDFVolumeProcessorCuda::ScalableTSDFVolumeProcessorCuda() {
    N_ = -1;
    max_subvolumes_ = -1;
}


ScalableTSDFVolumeProcessorCuda::ScalableTSDFVolumeProcessorCuda(
    int N, int max_subvolumes) {
    Create(N, max_subvolumes);
}


ScalableTSDFVolumeProcessorCuda::ScalableTSDFVolumeProcessorCuda(
    const ScalableTSDFVolumeProcessorCuda &other) {
    N_ = other.N_;
    max_subvolumes_ = other.max_subvolumes_;

    device_ = other.device_;
}


ScalableTSDFVolumeProcessorCuda &ScalableTSDFVolumeProcessorCuda::operator=(
    const ScalableTSDFVolumeProcessorCuda &other) {
    if (this != &other) {
        N_ = other.N_;
        max_subvolumes_ = other.max_subvolumes_;

        device_ = other.device_;
    }
    return *this;
}


ScalableTSDFVolumeProcessorCuda::~ScalableTSDFVolumeProcessorCuda() {
    Release();
}


void ScalableTSDFVolumeProcessorCuda::Create(
    int N, int max_subvolumes) {
    if (device_ != nullptr) {
        utility::PrintError("[ScalableGradientVolumeCuda]: "
                            "Already created, abort!\n");
        return;
    }

    assert(N_ > 0 && max_subvolumes > 0);

    device_ = std::make_shared<ScalableTSDFVolumeProcessorCudaDevice>();

    N_ = N;
    max_subvolumes_ = max_subvolumes;

    const int NNN = N_ * N_ * N_;
    CheckCuda(cudaMalloc(&device_->gradient_memory_pool_,
                         sizeof(Vector3f) * NNN * max_subvolumes_));

    UpdateDevice();
    Reset();
}


void ScalableTSDFVolumeProcessorCuda::Release() {
    if (device_ != nullptr && device_.use_count() == 1) {
        CheckCuda(cudaFree(device_->gradient_memory_pool_));
    }

    device_ = nullptr;
    max_subvolumes_ = -1;
}


void ScalableTSDFVolumeProcessorCuda::Reset() {
    if (device_ != nullptr) {
        const size_t NNN = N_ * N_ * N_;
        CheckCuda(cudaMemset(device_->gradient_memory_pool_, 0,
                             sizeof(Vector3f) * NNN * max_subvolumes_));
    }
}


void ScalableTSDFVolumeProcessorCuda::UpdateDevice() {
    if (device_ != nullptr) {
        device_->N_ = N_;
    }
}

void ScalableTSDFVolumeProcessorCuda::ComputeGradient(ScalableTSDFVolumeCuda &tsdf_volume){
    assert(device_ != nullptr);

    active_subvolumes_ = tsdf_volume.active_subvolume_entry_array_.size();
    utility::PrintDebug("Active subvolumes: %d\n", active_subvolumes_);

    if (active_subvolumes_ <= 0) {
        utility::PrintError("Invalid active subvolume numbers: %d !\n",
                   active_subvolumes_);
        return;
    }

    ScalableGradientVolumeCudaKernelCaller::ComputeGradient(*this, tsdf_volume);
}

PointCloudCuda ScalableTSDFVolumeProcessorCuda::ExtractVoxelsNearSurface(
    ScalableTSDFVolumeCuda &tsdf_volume,
    float threshold) {

    PointCloudCuda pcl(VertexWithNormalAndColor,
                       tsdf_volume.active_subvolume_entry_array_.size()
                       * (N_ * N_ * N_));
    pcl.points_.set_iterator(0);
    ScalableGradientVolumeCudaKernelCaller::ExtractVoxelsNearSurfaces(
        *this, tsdf_volume, pcl, threshold);
    pcl.colors_.set_iterator(pcl.points_.size());
    pcl.normals_.set_iterator(pcl.points_.size());

    return pcl;
}
} // cuda
} // open3d