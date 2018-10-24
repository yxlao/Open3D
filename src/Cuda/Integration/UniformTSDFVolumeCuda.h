//
// Created by wei on 10/9/18.
//

#pragma once

#include "IntegrationClasses.h"
#include <Cuda/Geometry/VectorCuda.h>
#include <Cuda/Geometry/ImageCuda.h>
#include <Cuda/Geometry/PinholeCameraCuda.h>
#include <Cuda/Geometry/TransformCuda.h>
#include <Cuda/Geometry/TriangleMeshCuda.h>

#include <cstdlib>
#include <memory>

namespace open3d {

template<size_t N>
class __ALIGN__(16) UniformTSDFVolumeCudaServer {
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
    __DEVICE__ inline int IndexOf(int x, int y, int z) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(x >= 0 && y >= 0 && z >= 0);
        assert(x < N && y < N && z < N);
#endif
        return int(z * (N * N) + y * N + x);
    }
    __DEVICE__ inline int IndexOf(const Vector3i &X) {
        return IndexOf(X(0), X(1), X(2));
    }

public:
    /** Direct index accessing
      * - for efficiency ignore index checking in these functions
      * - check them outside **/
    __DEVICE__ inline float &tsdf(int x, int y, int z) {
        return tsdf_[IndexOf(x, y, z)];
    }
    __DEVICE__ inline float &tsdf(const Vector3i &X) {
        return tsdf_[IndexOf(X(0), X(1), X(2))];
    }
    __DEVICE__ inline uchar &weight(int x, int y, int z) {
        return weight_[IndexOf(x, y, z)];
    }
    __DEVICE__ inline uchar &weight(const Vector3i &X) {
        return weight_[IndexOf(X(0), X(1), X(2))];
    }
    __DEVICE__ inline Vector3b &color(int x, int y, int z) {
        return color_[IndexOf(x, y, z)];
    }
    __DEVICE__ inline Vector3b &color(const Vector3i &X) {
        return color_[IndexOf(X(0), X(1), X(2))];
    }

    /** Voxel level trivial gradient -- NO trilinear interpolation
     * This is especially useful for MarchingCubes **/
    __DEVICE__ Vector3f gradient(int x, int y, int z);
    __DEVICE__ Vector3f gradient(const Vector3i &X);


    /** Coordinate conversions **/
    __DEVICE__ inline bool InVolume(int x, int y, int z);
    __DEVICE__ inline bool InVolume(const Vector3i &X);

    __DEVICE__ inline bool InVolumef(float x, float y, float z);
    __DEVICE__ inline bool InVolumef(const Vector3f &X);

    __DEVICE__ inline Vector3f world_to_voxel(float xw, float yw, float zw);
    __DEVICE__ inline Vector3f world_to_voxel(const Vector3f &Xw);

    __DEVICE__ inline Vector3f voxel_to_world(float x, float y, float z);
    __DEVICE__ inline Vector3f voxel_to_world(const Vector3f &X);

    __DEVICE__ inline Vector3f voxel_to_volume(const Vector3f &X);
    __DEVICE__ inline Vector3f voxel_to_volume(float x, float y, float z);

    __DEVICE__ inline Vector3f volume_to_voxel(const Vector3f &Xv);
    __DEVICE__ inline Vector3f volume_to_voxel(float xv, float yv, float zv);

public:
    /** Value interpolating **/
    __DEVICE__ float TSDFAt(float x, float y, float z);
    __DEVICE__ float TSDFAt(const Vector3f &X);

    __DEVICE__ uchar WeightAt(float x, float y, float z);
    __DEVICE__ uchar WeightAt(const Vector3f &X);

    __DEVICE__ Vector3b ColorAt(float x, float y, float z);
    __DEVICE__ Vector3b ColorAt(const Vector3f &X);

    __DEVICE__ Vector3f GradientAt(float x, float y, float z);
    __DEVICE__ Vector3f GradientAt(const Vector3f &X);

public:
    /** WARNING!!! DO NOT USE IT!!!
      * This method is reserved for ScalableTSDFVolumeCudaServer
      * That class requires us to initialize memory ON GPU. */
    __DEVICE__ void Create(float *tsdf, uchar *weight, Vector3b *color);

public:
    __DEVICE__ void Integrate(int x, int y, int z,
                              ImageCudaServer<Vector1f> &depth,
                              MonoPinholeCameraCuda &camera,
                              TransformCuda &transform_camera_to_world);

    __DEVICE__ Vector3f RayCasting(int x, int y,
                                   MonoPinholeCameraCuda &camera,
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
    void Integrate(ImageCuda<Vector1f> &depth,
                   MonoPinholeCameraCuda &camera,
                   TransformCuda &transform_camera_to_world);
    void RayCasting(ImageCuda<Vector3f> &image,
                    MonoPinholeCameraCuda &camera,
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
                     ImageCudaServer<Vector1f> depth,
                     MonoPinholeCameraCuda camera,
                     TransformCuda transform_camera_to_world);

template<size_t N>
__GLOBAL__
void RayCastingKernel(UniformTSDFVolumeCudaServer<N> server,
                      ImageCudaServer<Vector3f> image,
                      MonoPinholeCameraCuda camera,
                      TransformCuda transform_camera_to_world);
}