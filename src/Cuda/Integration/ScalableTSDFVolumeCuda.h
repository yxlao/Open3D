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
public:
    typedef HashTableCudaServer<Vector3i, UniformTSDFVolumeCudaServer<N>,
                                SpatialHasher> SpatialHashTableCudaServer;

public:
    int bucket_count_;
    int value_capacity_;

    float voxel_length_;
    float inv_voxel_length_;
    float sdf_trunc_;
    TransformCuda transform_volume_to_world_;
    TransformCuda transform_world_to_volume_;

private:
    SpatialHashTableCudaServer hash_table_;

    float *tsdf_memory_pool_;
    uchar *weight_memory_pool_;
    Vector3b *color_memory_pool_;

public:
    __HOSTDEVICE__ inline SpatialHashTableCudaServer &hash_table() {
        return hash_table_;
    }
    __DEVICE__ inline float *tsdf_memory_pool() {
        return tsdf_memory_pool_;
    }
    __DEVICE__ inline uchar *weight_memory_pool() {
        return weight_memory_pool_;
    }
    __DEVICE__ inline Vector3b *color_memory_pool() {
        return color_memory_pool_;
    }

public:
    friend class ScalableTSDFVolumeCuda<N>;
};

template<size_t N>
class ScalableTSDFVolumeCuda {
public:
    /** Note here the template is exactly the same as the
     * SpatialHashTableCudaServer.
     * We will explicitly deal with the UniformTSDFVolumeCudaServer later
     * **/
    typedef HashTableCuda
        <Vector3i, UniformTSDFVolumeCudaServer<N>, SpatialHasher>
        SpatialHashTableCuda;

private:
    std::shared_ptr<ScalableTSDFVolumeCudaServer<N>> server_ = nullptr;
    SpatialHashTableCuda hash_table_;

public:
    int bucket_count_;
    int value_capacity_;

    float voxel_length_;
    float sdf_trunc_;
    TransformCuda transform_volume_to_world_;

public:
    ScalableTSDFVolumeCuda();
    ScalableTSDFVolumeCuda(int bucket_count, int value_capacity,
                           float voxel_length, float sdf_trunc,
                           TransformCuda &transform_volume_to_world = TransformCuda::Identity());
    ScalableTSDFVolumeCuda(const ScalableTSDFVolumeCuda<N> &other);
    ScalableTSDFVolumeCuda<N> &operator=(const ScalableTSDFVolumeCuda<N> &other);
    ~ScalableTSDFVolumeCuda();

    /** BE CAREFUL, we have to rewrite some
     * non-wrapped allocation stuff here for UniformTSDFVolumeCudaServer **/
    void Create(int bucket_count, int value_capacity);
    void Release();
    void UpdateServer();

    std::pair<std::vector<Vector3i>,          /* Keys */
              std::vector<std::tuple<std::vector<float>,  /* TSDF volumes */
                                     std::vector<uchar>,
                                     std::vector<Vector3b>>>> DownloadVolumes();

    void Reset();

public:
    SpatialHashTableCuda &hash_table() {
        return hash_table_;
    }
    const SpatialHashTableCuda &hash_table() const {
        return hash_table_;
    }
    std::shared_ptr<ScalableTSDFVolumeCudaServer<N>> &server() {
        return server_;
    }
    const std::shared_ptr<ScalableTSDFVolumeCudaServer<N>> &
    server() const {
        return server_;
    }
};

template<size_t N>
__GLOBAL__
void CreateScalableTSDFVolumesKernel(ScalableTSDFVolumeCudaServer<N> server);
}
