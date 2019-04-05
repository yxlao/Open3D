//
// Created by wei on 10/23/18.
//

#pragma once

#include <Cuda/Common/UtilsCuda.h>

#include <Cuda/Container/ArrayCuda.h>
#include <Cuda/Geometry/TriangleMeshCuda.h>
#include <Cuda/Common/LinearAlgebraCuda.h>

#include <memory>
#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>

namespace open3d {
namespace cuda {

class ScalableTSDFVolumeProcessorCudaDevice {
public:
    Vector3f *gradient_memory_pool_;

public:
    int N_;

    __DEVICE__ inline int IndexOf(const Vector3i &Xlocal,
                                  int subvolume_idx) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(Xlocal(0) >= 0 && Xlocal(1) >= 0 && Xlocal(2) >= 0 &&
            subvolume_idx >= 0);
        assert(Xlocal(0) < N && Xlocal(1) < N && Xlocal(2) < N);
#endif
        return int(Xlocal(2) * (N_ * N_) + Xlocal(1) * N_ + Xlocal(0)
                       + subvolume_idx * (N_ * N_ * N_));
    }

    __DEVICE__ inline Vector3f &gradient(
        const Vector3i &Xlocal, int subvolume_idx) {
        return gradient_memory_pool_[IndexOf(Xlocal, subvolume_idx)];
    }
};

class ScalableTSDFVolumeProcessorCuda {
public:
    std::shared_ptr<ScalableTSDFVolumeProcessorCudaDevice> device_ = nullptr;

public:
    int N_;

    int active_subvolumes_;
    int max_subvolumes_;

public:
    ScalableTSDFVolumeProcessorCuda();
    ScalableTSDFVolumeProcessorCuda(int N, int max_subvolumes);
    ScalableTSDFVolumeProcessorCuda(const ScalableTSDFVolumeProcessorCuda &other);
    ScalableTSDFVolumeProcessorCuda &operator=(const ScalableTSDFVolumeProcessorCuda &other);
    ~ScalableTSDFVolumeProcessorCuda();

    void Create(int N, int max_subvolumes);
    void Release();
    void Reset();
    void UpdateDevice();

public:
    void ComputeGradient(ScalableTSDFVolumeCuda &tsdf_volume);
    PointCloudCuda ExtractVoxelsNearSurface(ScalableTSDFVolumeCuda &tsdf_volume,
                                            float threshold);
};

class ScalableGradientVolumeCudaKernelCaller {
public:
    static void ComputeGradient(ScalableTSDFVolumeProcessorCuda &server,
                                ScalableTSDFVolumeCuda &tsdf_volume);
    static void ExtractVoxelsNearSurfaces(ScalableTSDFVolumeProcessorCuda &server,
                                          ScalableTSDFVolumeCuda &volume,
                                          PointCloudCuda &pcl,
                                          float threshold);
};

__GLOBAL__
void ComputeGradientKernel(ScalableTSDFVolumeProcessorCudaDevice server,
                           ScalableTSDFVolumeCudaDevice tsdf_volume);
__GLOBAL__
void ExtractVoxelsNearSurfaceKernel(ScalableTSDFVolumeProcessorCudaDevice server,
                                    ScalableTSDFVolumeCudaDevice volume,
                                    PointCloudCudaDevice pcl,
                                    float threshold);
} // cuda
} // open3d