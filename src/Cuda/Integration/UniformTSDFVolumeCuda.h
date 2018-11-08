//
// Created by wei on 10/9/18.
//

#pragma once

#include "IntegrationClasses.h"

#include <Cuda/Geometry/ImageCuda.h>
#include <Cuda/Camera/PinholeCameraIntrinsicCuda.h>
#include <Cuda/Geometry/RGBDImageCuda.h>
#include <Cuda/Geometry/TransformCuda.h>
#include <Cuda/Geometry/TriangleMeshCuda.h>
#include <Cuda/Geometry/VectorCuda.h>

#include <cstdlib>
#include <memory>

namespace open3d {

template<size_t N>
class UniformTSDFVolumeCudaServer {
private:
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
        ret(0) = int(index % N);
        ret(1) = int((index % (N * N)) / N);
        ret(2) = int(index / (N * N));
        return ret;
    }
    __DEVICE__ inline int IndexOf(const Vector3i &X) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(X(0) >= 0 && X(1) >= 0 && X(2) >= 0);
        assert(X(0) < N && X(1) < N && X(2) < N);
#endif
        return int(X(2) * (N * N) + X(1) * N + X(0));
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
    /** WARNING!!! DO NOT USE IT!!!
      * This method is reserved for ScalableTSDFVolumeCudaServer
      * That class requires us to initialize memory ON GPU. */
    __DEVICE__ void Create(float *tsdf, uchar *weight, Vector3b *color);

public:
    __DEVICE__ void Integrate(const Vector3i &X,
                              RGBDImageCudaServer &rgbd,
                              PinholeCameraIntrinsicCuda &camera,
                              TransformCuda &transform_camera_to_world);

    __DEVICE__ Vector3f RayCasting(const Vector2i &p,
                                   PinholeCameraIntrinsicCuda &camera,
                                   TransformCuda &transform_camera_to_world);

public:
    friend class UniformTSDFVolumeCuda<N>;
    friend class ScalableTSDFVolumeCuda<N>;
};

template<size_t N>
class UniformTSDFVolumeCuda {
private:
    std::shared_ptr<UniformTSDFVolumeCudaServer<N>> server_ = nullptr;

public:
    float voxel_length_;
    float sdf_trunc_;
    TransformCuda transform_volume_to_world_;

public:
    UniformTSDFVolumeCuda();
    UniformTSDFVolumeCuda(float voxel_length, float sdf_trunc,
                          TransformCuda &volume_to_world);
    UniformTSDFVolumeCuda(const UniformTSDFVolumeCuda<N> &other);
    UniformTSDFVolumeCuda<N> &operator=(const UniformTSDFVolumeCuda<N> &other);
    ~UniformTSDFVolumeCuda();

    void Create();
    void Release();
    void UpdateServer();

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
    void RayCasting(ImageCuda<Vector3f> &image,
                    PinholeCameraIntrinsicCuda &camera,
                    TransformCuda &transform_camera_to_world);

public:
    std::shared_ptr<UniformTSDFVolumeCudaServer<N>> &server() {
        return server_;
    }
    const std::shared_ptr<UniformTSDFVolumeCudaServer<N>> &server() const {
        return server_;
    }
};

template<size_t N>
__GLOBAL__
void IntegrateKernel(UniformTSDFVolumeCudaServer<N> server,
                     RGBDImageCudaServer depth,
                     PinholeCameraIntrinsicCuda camera,
                     TransformCuda transform_camera_to_world);

template<size_t N>
__GLOBAL__
void RayCastingKernel(UniformTSDFVolumeCudaServer<N> server,
                      ImageCudaServer<Vector3f> image,
                      PinholeCameraIntrinsicCuda camera,
                      TransformCuda transform_camera_to_world);
}