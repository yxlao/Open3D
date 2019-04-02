//
// Created by wei on 10/9/18.
//

#pragma once

#include "IntegrationClasses.h"

#include <Cuda/Geometry/ImageCuda.h>
#include <Cuda/Camera/PinholeCameraIntrinsicCuda.h>
#include <Cuda/Geometry/RGBDImageCuda.h>
#include <Cuda/Common/TransformCuda.h>
#include <Cuda/Geometry/TriangleMeshCuda.h>
#include <Cuda/Common/LinearAlgebraCuda.h>

#include <cstdlib>
#include <memory>

namespace open3d {
namespace cuda {

class UniformTSDFVolumeCudaDevice {
public:
    int N_;

    /** [N * N * N] **/
    float *tsdf_;
    uchar *weight_;
    Vector3b *color_;

public:
    /** According to UniformTSDFVolume.cpp,
     * Voxel xyz is at Vector3f(0.5) + [x, y, z]^T * voxel_length_;
     * Shared parameters with UniformTDFVolumes
     **/
    float voxel_length_;
    float inv_voxel_length_;
    float sdf_trunc_;
    TransformCuda transform_volume_to_world_;
    TransformCuda transform_world_to_volume_;

public:
    __DEVICE__ inline Vector3i Vectorize(size_t index) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(index < N * N * N);
#endif
        Vector3i ret;
        ret(0) = int(index % N_);
        ret(1) = int((index % (N_ * N_)) / N_);
        ret(2) = int(index / (N_ * N_));
        return ret;
    }
    __DEVICE__ inline int IndexOf(const Vector3i &X) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(X(0) >= 0 && X(1) >= 0 && X(2) >= 0);
        assert(X(0) < N && X(1) < N && X(2) < N);
#endif
        return int(X(2) * (N_ * N_) + X(1) * N_ + X(0));
    }

public:
    /** Direct index accessing
      * - for efficiency ignore index checking in these functions
      * - check them outside **/
    __DEVICE__ inline float &tsdf(const Vector3i &X) {
        return tsdf_[IndexOf(X)];
    }
    __DEVICE__ inline uchar &weight(const Vector3i &X) {
        return weight_[IndexOf(X)];
    }
    __DEVICE__ inline Vector3b &color(const Vector3i &X) {
        return color_[IndexOf(X)];
    }

    /** Voxel level gradient -- NO trilinear interpolation.
     * This is especially useful for MarchingCubes **/
    __DEVICE__ Vector3f gradient(const Vector3i &X);

    /** Coordinate conversions **/
    __DEVICE__ inline bool InVolume(const Vector3i &X);
    __DEVICE__ inline bool InVolumef(const Vector3f &X);

    __DEVICE__ inline Vector3f world_to_voxelf(const Vector3f &Xw);
    __DEVICE__ inline Vector3f voxelf_to_world(const Vector3f &X);
    __DEVICE__ inline Vector3f volume_to_voxelf(const Vector3f &Xv);
    __DEVICE__ inline Vector3f voxelf_to_volume(const Vector3f &X);

public:
    /** Value interpolating **/
    __DEVICE__ float TSDFAt(const Vector3f &X);
    __DEVICE__ uchar WeightAt(const Vector3f &X);
    __DEVICE__ Vector3b ColorAt(const Vector3f &X);
    __DEVICE__ Vector3f GradientAt(const Vector3f &X);

public:
    __DEVICE__ void Integrate(const Vector3i &X,
                              RGBDImageCudaDevice &rgbd,
                              PinholeCameraIntrinsicCuda &camera,
                              TransformCuda &transform_camera_to_world);

    __DEVICE__ Vector3f RayCasting(const Vector2i &p,
                                   PinholeCameraIntrinsicCuda &camera,
                                   TransformCuda &transform_camera_to_world);

public:
    friend class UniformTSDFVolumeCuda;
    friend class ScalableTSDFVolumeCuda;
};


class UniformTSDFVolumeCuda {
public:
    std::shared_ptr<UniformTSDFVolumeCudaDevice> device_ = nullptr;

public:
    int N_;

    float voxel_length_;
    float sdf_trunc_;
    TransformCuda transform_volume_to_world_;

public:
    UniformTSDFVolumeCuda();
    UniformTSDFVolumeCuda(int N, float voxel_length, float sdf_trunc,
                          TransformCuda &volume_to_world);
    UniformTSDFVolumeCuda(const UniformTSDFVolumeCuda &other);
    UniformTSDFVolumeCuda &operator=(const UniformTSDFVolumeCuda &other);
    ~UniformTSDFVolumeCuda();

    void Create(int N);
    void Release();
    void UpdateDevice();

    void Reset();

    void UploadVolume(std::vector<float> &tsdf,
                      std::vector<uchar> &weight,
                      std::vector<Vector3b> &color);
    std::tuple<std::vector<float>, std::vector<uchar>, std::vector<Vector3b>>
    DownloadVolume();

public:
    void Integrate(RGBDImageCuda &rgbd,
                   PinholeCameraIntrinsicCuda &camera,
                   TransformCuda &transform_camera_to_world);
    void RayCasting(ImageCuda<float, 3> &image,
                    PinholeCameraIntrinsicCuda &camera,
                    TransformCuda &transform_camera_to_world);
};


class UniformTSDFVolumeCudaKernelCaller {
public:
    static void Integrate(
        UniformTSDFVolumeCuda &volume,
        RGBDImageCuda &rgbd,
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world);

    static void RayCasting(
        UniformTSDFVolumeCuda &volume,
        ImageCuda<float, 3> &image,
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world);
};


__GLOBAL__
void IntegrateKernel(UniformTSDFVolumeCudaDevice server,
                     RGBDImageCudaDevice depth,
                     PinholeCameraIntrinsicCuda camera,
                     TransformCuda transform_camera_to_world);


__GLOBAL__
void RayCastingKernel(UniformTSDFVolumeCudaDevice server,
                      ImageCudaDevice<float, 3> image,
                      PinholeCameraIntrinsicCuda camera,
                      TransformCuda transform_camera_to_world);
} // cuda
} // open3d