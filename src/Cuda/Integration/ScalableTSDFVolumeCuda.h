//
// Created by wei on 10/10/18.
//

#pragma once

#include <Cuda/Common/UtilsCuda.h>
#include <Cuda/Container/HashTableCuda.h>
#include <Cuda/Geometry/TransformCuda.h>
#include <Cuda/Geometry/VectorCuda.h>
#include <Cuda/Geometry/PinholeCameraCuda.h>
#include "IntegrationClasses.h"

#include <memory>

namespace open3d {

template<size_t N>
class __ALIGN__(16) ScalableTSDFVolumeCudaServer {

private: /** [N * N * N]; **/
    HashTableCudaServer
        <Vector3i, UniformTSDFVolumeCudaServer<N>, SpatialHasher> hash_table_;

public:
    float voxel_length_;
    float inv_voxel_length_;
    float sdf_trunc_;
    TransformCuda transform_volume_to_world_;
    TransformCuda transform_world_to_volume_;

public: /** Conversions **/
    /** Voxel coordinate can be arbitrary value.
      * They can be e.g. (1230, 1024, -1024). We then convert them in the
      * desired block to access them if they exist. **/
    inline __DEVICE__ Vector3f world_to_voxel(float x, float y, float z);
    inline __DEVICE__ Vector3f world_to_voxel(const Vector3f &X);

    inline __DEVICE__ Vector3f voxel_to_world(float x, float y, float z);
    inline __DEVICE__ Vector3f voxel_to_world(const Vector3f &X);

    inline __DEVICE__ Vector3f voxel_to_volume(float x, float y, float z);
    inline __DEVICE__ Vector3f voxel_to_volume(const Vector3f &X);

    inline __DEVICE__ Vector3f volume_to_voxel(float x, float y, float z);
    inline __DEVICE__ Vector3f volume_to_voxel(const Vector3f &X);

    inline __DEVICE__ Vector3i voxel_to_block(float x, float y, float z);
    inline __DEVICE__ Vector3i voxel_to_block(const Vector3f &X);

    inline __DEVICE__ Vector3i voxel_in_block(
        float x, float y, float z, int block_x, int block_y, int block_z);
    inline __DEVICE__ Vector3i voxel_in_block(
        const Vector3f &X, const Vector3i &block);

public:
    /** Direct index accessing are wrapped in UniformVolumes
     * Find the block with voxel_to_block, and access it explicitly.
     */
     inline __DEVICE__ float tsdf(int x, int y, int z);
     inline __DEVICE__ float tsdf(const Vector3i &X);
     inline __DEVICE__ uchar weight(int x, int y, int z);
     inline __DEVICE__ uchar weight(const Vector3i &X);
     inline __DEVICE__ Vector3b color(int x, int y, int z);
     inline __DEVICE__ Vector3b color(const Vector3i &X);

     inline __DEVICE__ Vector3f gradient(int x, int y, int z);
     inline __DEVICE__ Vector3f gradient(const Vector3i &X);

public:
    /** Value interpolating (across sub-volumes)
     * DO NOT frequently use them. They can be really slow.
     * (TRY TO) ONLY use them for boundary value interpolations. **/
    inline __DEVICE__ float TSDFAt(float x, float y, float z);
    inline __DEVICE__ float TSDFAt(const Vector3f &X);

    inline __DEVICE__ uchar WeightAt(float x, float y, float z);
    inline __DEVICE__ uchar WeightAt(const Vector3f &X);

    inline __DEVICE__ Vector3b ColorAt(float x, float y, float z);
    inline __DEVICE__ Vector3b ColorAt(const Vector3f &X);

    inline __DEVICE__ Vector3f GradientAt(float x, float y, float z);
    inline __DEVICE__ Vector3f GradientAt(const Vector3f &X);

public:
    __DEVICE__ void Integrate(
        int x, int y, int z,
        ImageCudaServer<Vector1f> &depth,
        MonoPinholeCameraCuda &camera,
        TransformCuda &transform_camera_to_world);

    __DEVICE__ Vector3f
    RayCasting(
        int x, int y,
        MonoPinholeCameraCuda &camera,
        TransformCuda &transform_camera_to_world);

public:
    friend class ScalableTSDFVolumeCuda<N>;
};

template<size_t N>
class ScalableTSDFVolumeCuda {
private:
    std::shared_ptr<ScalableTSDFVolumeCudaServer<N>> server_ = nullptr;

public:
    float voxel_length_;
    float sdf_trunc_;
    TransformCuda transform_volume_to_world_;

public:
    ScalableTSDFVolumeCuda();
    ScalableTSDFVolumeCuda(float voxel_length, float sdf_trunc,
                           TransformCuda &volume_to_world = TransformCuda::Identity());
    ScalableTSDFVolumeCuda(const ScalableTSDFVolumeCuda<N> &other);
    ScalableTSDFVolumeCuda<N> &operator=(const ScalableTSDFVolumeCuda<N> &other);
    ~ScalableTSDFVolumeCuda();

    /** BE CAREFUL, we have to rewrite some
     * non-wrapped allocation stuff here for UniformTSDFVolumeCudaServer **/
    void Create();
    void Release();
    void UpdateServer();

    void Reset();

    void UploadVolume(std::vector<float> &tsdf, std::vector<uchar> &weight,
                      std::vector<Vector3b> &color);
    std::tuple<std::vector<float>, std::vector<uchar>, std::vector<Vector3b>>
    DownloadVolume();

    int Integrate(ImageCuda<Vector1f> &depth,
                  MonoPinholeCameraCuda &camera,
                  TransformCuda &transform_camera_to_world);

    template<VertexType type>
    int MarchingCubes(UniformMeshVolumeCuda<type, N> &mesher);

    int RayCasting(ImageCuda<Vector3f> &image,
                   MonoPinholeCameraCuda &camera,
                   TransformCuda &transform_camera_to_world);

public:
    std::shared_ptr<ScalableTSDFVolumeCudaServer<N>> &
    server() {
        return server_;
    }
    const std::shared_ptr<ScalableTSDFVolumeCudaServer<N>> &
    server() const {
        return server_;
    }
};

template<size_t N>
__GLOBAL__
void IntegrateKernel(ScalableTSDFVolumeCudaServer<N> server,
                     ImageCudaServer<Vector1f> depth,
                     MonoPinholeCameraCuda camera,
                     TransformCuda transform_camera_to_world);

template<size_t N>
__GLOBAL__
void RayCastingKernel(ScalableTSDFVolumeCudaServer<N> server,
                      ImageCudaServer<Vector3f> image,
                      MonoPinholeCameraCuda camera,
                      TransformCuda transform_camera_to_world);

}
