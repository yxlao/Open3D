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
    typedef HashTableCudaServer<
        Vector3i, UniformTSDFVolumeCudaServer<N>, SpatialHasher>
        SpatialHashTableCudaServer;

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

    /** An extension to hash_table_.assigned_entries_array_ **/
    ArrayCudaServer<HashEntry<Vector3i>> target_subvolume_entry_array_;

    float *tsdf_memory_pool_;
    uchar *weight_memory_pool_;
    Vector3b *color_memory_pool_;

public:
    __HOSTDEVICE__ inline SpatialHashTableCudaServer &hash_table() {
        return hash_table_;
    }
    __HOSTDEVICE__ inline ArrayCudaServer<HashEntry<Vector3i>>&
    target_subvolume_entry_array() {
        return target_subvolume_entry_array_;
    }
    __HOSTDEVICE__ inline float *tsdf_memory_pool() {
        return tsdf_memory_pool_;
    }
    __HOSTDEVICE__ inline uchar *weight_memory_pool() {
        return weight_memory_pool_;
    }
    __HOSTDEVICE__ inline Vector3b *color_memory_pool() {
        return color_memory_pool_;
    }

public:
    /** Coordinate conversions
     *  Duplicate functions of UniformTSDFVolume (how to simplify?) **/
    __DEVICE__ inline Vector3f world_to_voxel(float xw, float yw, float zw);
    __DEVICE__ inline Vector3f world_to_voxel(const Vector3f &Xw);

    __DEVICE__ inline Vector3f voxel_to_world(float x, float y, float z);
    __DEVICE__ inline Vector3f voxel_to_world(const Vector3f &X);

    __DEVICE__ inline Vector3f voxel_to_volume(const Vector3f &X);
    __DEVICE__ inline Vector3f voxel_to_volume(float x, float y, float z);

    __DEVICE__ inline Vector3f volume_to_voxel(const Vector3f &Xv);
    __DEVICE__ inline Vector3f volume_to_voxel(float xv, float yv, float zv);

    /**
     * NOTE: to access voxel values, we DO NOT DEEPLY COUPLE FUNCTIONS.
     * Otherwise there will be too much redundant operations.
     *
     * Given (x, y, z) in voxel units, the query process should be
     * 1. > Get volume index it is in
     *      Vector3i subvolume_idx = subvolume_index[f](x, y, z):
     * 2. > Get volume pointer (server)
     *      UniformTSDFVolumeCudaServer* subvolume = query_subvolume(subvolume_idx);
     *      if (subvolume == nullptr) break or return;
     * 3. > Get voxel coordinate in this specific volume
     *      Vector3f offset = subvolume_offset(x, y, z, subvolume_idx);
     * 4. > Access or interpolate in this volume (and neighbors)
     *      int -> direct access with subvolume->tsdf, etc
     *      float -> we should turn to interpolations
     **/
    /** Similar to LocateVolumeUnit **/
    __DEVICE__ inline Vector3i voxel_locate_subvolume(int x, int y, int z);
    __DEVICE__ inline Vector3i voxel_locate_subvolume(const Vector3i &X);

    __DEVICE__ inline Vector3i voxelf_locate_subvolume(
        float x, float y, float z);
    __DEVICE__ inline Vector3i voxelf_locate_subvolume(
        const Vector3f &X);

    __DEVICE__ inline Vector3i voxel_global_to_local(
        int x, int y, int z, const Vector3i &Xsv);
    __DEVICE__ inline Vector3i voxel_global_to_local(
        const Vector3i &X, const Vector3i &Xsv);

    __DEVICE__ inline Vector3f voxelf_global_to_local(
        float x, float y, float z, const Vector3i &Xsv);
    __DEVICE__ inline Vector3f voxelf_global_to_local(
        const Vector3f &X, const Vector3i &Xsv);

    __DEVICE__ inline Vector3i voxel_local_to_global(
        int xlocal, int ylocal, int zlocal, const Vector3i &Xsv);
    __DEVICE__ inline Vector3i voxel_local_to_global(
        const Vector3i &Xlocal, const Vector3i &Xsv);

    __DEVICE__ inline Vector3f voxelf_local_to_global(
        float xlocal, float ylocal, float zlocal, const Vector3i &Xsv);
    __DEVICE__ inline Vector3f voxelf_local_to_global(
        const Vector3f &Xlocal, const Vector3i &Xsv);

    /** Note when we assume we already know @offset is in @block.
     *  For interpolation, boundary regions are [N-1~N] for float, none for int
     *  For gradient, boundary regions are
     *  > [N-2 ~ N) and [0 ~ 1)
     * **/
    __DEVICE__ inline bool OnBoundary(
        int xlocal, int ylocal, int zlocal, bool for_gradient = false);
    __DEVICE__ inline bool OnBoundary(
        const Vector3i &Xlocal, bool for_gradient = false);
    __DEVICE__ inline bool OnBoundaryf(
        float xlocal, float ylocal, float zlocal, bool for_gradinet = false);
    __DEVICE__ inline bool OnBoundaryf(
        const Vector3f &Xlocal, bool for_gradient = false);

    /**
     * NOTE: interpolation, especially on boundaries, is very expensive.
     * To interpolate, kernels will frequently query neighbor volumes.
     * In a typical kernel, we should pre-store them in __shared__ memory.
     * __shared__ UniformTSDFVolumeCudaServer<N>* subvolume_neighbors[N];
     *
     * > For value interpolation, we define 8 neighbor subvolume indices as
     *  (0, 1) x (0, 1) x (0, 1)
     *
     * > For gradient interpolation, we define 20 neighbor indices as
     *   (w.r.t. smallest coordinate)
     *   (0, 1) x (0, 1) x (0, 1)   8
     *   (-1, ) x (0, 1) x (0, 1) + 4
     *   (0, 1) x (-1, ) x (0, 1) + 4
     *   (0, 1) x (0, 1) x (-1, ) + 4
     *
     * > To simplify, we define all the 27 neighbor indices
     *   (not necessary to assign them all in pre-processing).
     *   (-1, 0, 1) ^ 3
     */
    __DEVICE__ UniformTSDFVolumeCudaServer<N> *QuerySubvolume(
        const Vector3i &Xsv);

    __DEVICE__ void QuerySubvolumeWithNeighborIndex(
        const Vector3i &Xsv, int dxsv, int dysv, int dzsv,
        UniformTSDFVolumeCudaServer<N> **subvolume);

    __DEVICE__ inline Vector3i NeighborIndexOfBoundaryVoxel(
        int xlocal, int ylocal, int zlocal);
    __DEVICE__ inline Vector3i NeighborIndexOfBoundaryVoxel(
        const Vector3i &Xlocal);

    __DEVICE__ inline int LinearizeNeighborIndex(int dxsv, int dysv, int dzsv);
    __DEVICE__ inline int LinearizeNeighborIndex(const Vector3i &dXsv);

    /** In these functions range of input indices are [-1, N+1)
     * (xlocal, ylocal, zlocal) is inside
     * subvolumes[IndexOfNeighborSubvolumes(0, 0, 0)]
     **/
    /** Similar to uniform level gradient **/
    __DEVICE__ Vector3f gradient(
        int xlocal, int ylocal, int zlocal,
        UniformTSDFVolumeCudaServer<N> **subvolumes);
    __DEVICE__ Vector3f gradient(
        const Vector3i &Xlocal, UniformTSDFVolumeCudaServer<N> **subvolumes);

    __DEVICE__ float TSDFOnBoundaryAt(
        float xlocal, float ylocal, float zlocal,
        UniformTSDFVolumeCudaServer<N> **subvolumes);
    __DEVICE__ float TSDFOnBoundaryAt(
        const Vector3f &Xlocal, UniformTSDFVolumeCudaServer<N> **subvolumes);

    __DEVICE__ uchar WeightOnBoundaryAt(
        float xlocal, float ylocal, float zlocal,
        UniformTSDFVolumeCudaServer<N> **subvolumes);
    __DEVICE__ uchar WeightOnBoundaryAt(
        const Vector3f &Xlocal, UniformTSDFVolumeCudaServer<N> **subvolumes);

    __DEVICE__ Vector3b ColorOnBoundaryAt(
        float xlocal, float ylocal, float zlocal,
        UniformTSDFVolumeCudaServer<N> **subvolumes);
    __DEVICE__ Vector3b ColorOnBoundaryAt(
        const Vector3f &Xlocal, UniformTSDFVolumeCudaServer<N> **subvolumes);

    __DEVICE__ Vector3f GradientOnBoundaryAt(
        float xlocal, float ylocal, float zlocal,
        UniformTSDFVolumeCudaServer<N> **subvolumes);
    __DEVICE__ Vector3f GradientOnBoundaryAt(
        const Vector3f &Xlocal, UniformTSDFVolumeCudaServer<N> **subvolumes);

public:
    __DEVICE__ void TouchSubvolume(int x, int y,
                                   ImageCudaServer<Vector1f> &depth,
                                   MonoPinholeCameraCuda &camera,
                                   TransformCuda &transform_camera_to_world);
    __DEVICE__ void Integrate(int x, int y, int z,
                              ImageCudaServer<Vector1f> &depth,
                              MonoPinholeCameraCuda &camera,
                              TransformCuda &transform_camera_to_world);
    __DEVICE__ void RayCasting(int x, int y,
                               ImageCudaServer<Vector3b> &color,
                               MonoPinholeCameraCuda &camera,
                               TransformCuda &transform_camera_to_world);

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
    ArrayCuda<HashEntry<Vector3i>> target_subvolume_entry_array_;

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

    std::pair<std::vector<Vector3i>,                      /* Keys */
              std::vector<std::tuple<std::vector<float>,  /* TSDF volumes */
                                     std::vector<uchar>,
                                     std::vector<Vector3b>>>> DownloadVolumes();

public:
    void TouchBlocks(ImageCuda<Vector1f> &depth,
                     MonoPinholeCameraCuda &camera,
                     TransformCuda &transform_camera_to_world);
    void GetSubvolumesInFrustum(MonoPinholeCameraCuda &camera,
                            TransformCuda &transform_camera_to_world);
    void Integrate(ImageCuda<Vector1f> &depth,
                   MonoPinholeCameraCuda &camera,
                   TransformCuda &transform_camera_to_world);
    void RayCasting(ImageCuda<Vector1f> &image,
                    MonoPinholeCameraCuda &camera,
                    TransformCuda &transform_camera_to_world);

public:
    SpatialHashTableCuda &hash_table() {
        return hash_table_;
    }
    const SpatialHashTableCuda &hash_table() const {
        return hash_table_;
    }
    ArrayCuda<HashEntry<Vector3i>>& target_subvolume_entry_array() {
        return target_subvolume_entry_array_;
    }
    const ArrayCuda<HashEntry<Vector3i>>& target_subvolume_entry_array() const {
        return target_subvolume_entry_array_;
    }
    std::shared_ptr<ScalableTSDFVolumeCudaServer<N>> &server() {
        return server_;
    }
    const std::shared_ptr<ScalableTSDFVolumeCudaServer<N>> &server() const {
        return server_;
    }
};

template<size_t N>
__GLOBAL__
void CreateScalableTSDFVolumesKernel(ScalableTSDFVolumeCudaServer<N> server);

template<size_t N>
__GLOBAL__
void TouchSubvolumesKernel(ScalableTSDFVolumeCudaServer<N> server,
                       ImageCudaServer<Vector1f> depth,
                       MonoPinholeCameraCuda camera,
                       TransformCuda transform_camera_to_world);

template<size_t N>
__GLOBAL__
void IntegrateKernel(ScalableTSDFVolumeCudaServer<N> server,
                     ImageCudaServer<Vector1f> depth,
                     MonoPinholeCameraCuda camera,
                     TransformCuda transform_camera_to_world);
}
