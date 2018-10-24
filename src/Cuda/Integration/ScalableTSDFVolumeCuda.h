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

private:
    /** (N * N * N) * value_capacity **/
    float *tsdf_memory_pool_;
    uchar *weight_memory_pool_;
    Vector3b *color_memory_pool_;

    /** These are return values when subvolume is null.
     *  Refer to tsdf(), etc **/
    float tsdf_dummy_ = 0;
    uchar weight_dummy_ = 0;
    Vector3b color_dummy_ = Vector3b(0);

    SpatialHashTableCudaServer hash_table_;

    /** An extension to hash_table_.assigned_entries_array_ **/
    ArrayCudaServer<HashEntry<Vector3i>> target_subvolume_entry_array_;

public:
    int bucket_count_;
    int value_capacity_;

    float voxel_length_;
    float inv_voxel_length_;
    float sdf_trunc_;
    TransformCuda transform_volume_to_world_;
    TransformCuda transform_world_to_volume_;

public:
    __HOSTDEVICE__ inline SpatialHashTableCudaServer &hash_table() {
        return hash_table_;
    }
    __HOSTDEVICE__ inline ArrayCudaServer<HashEntry<Vector3i>> &
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

    __DEVICE__ UniformTSDFVolumeCudaServer<N> *QuerySubvolume(
        const Vector3i &Xsv);

    /** Unoptimized access and interpolation
     * (required hash-table access every access, good for RayCasting) **/
    __DEVICE__ float &tsdf(int x, int y, int z);
    __DEVICE__ float &tsdf(const Vector3i &X);

    __DEVICE__ uchar &weight(int x, int y, int z);
    __DEVICE__ uchar &weight(const Vector3i &X);

    __DEVICE__ Vector3b &color(int x, int y, int z);
    __DEVICE__ Vector3b &color(const Vector3i &X);

    __DEVICE__ float TSDFAt(float x, float y, float z);
    __DEVICE__ float TSDFAt(const Vector3f &X);

    __DEVICE__ uchar WeightAt(float x, float y, float z);
    __DEVICE__ uchar WeightAt(const Vector3f &X);

    __DEVICE__ Vector3b ColorAt(float x, float y, float z);
    __DEVICE__ Vector3b ColorAt(const Vector3f &X);

    __DEVICE__ Vector3f GradientAt(float x, float y, float z);
    __DEVICE__ Vector3f GradientAt(const Vector3f &X);

    /**
     * Optimized access and interpolation
     * (pre stored neighbors, good for MarchingCubes TBD)
     *
     * NOTE:
     * Interpolation, especially on boundaries, is very expensive when kernels
     * frequently query neighbor volumes.
     * In a typical kernel like MC, we pre-store them in __shared__ memory.
     *
     * In the beginning of a kernel, we pre-query these volumes and syncthreads.
     * The neighbors are stored in __shared__ subvolumes[N].
     *
     * > The neighbor subvolume indices for value interpolation are
     *  (0, 1) x (0, 1) x (0, 1) -> N = 8
     *
     * > For gradient interpolation, more neighbors have to be considered
     *   (0, 1) x (0, 1) x (0, 1)   8
     *   (-1, ) x (0, 1) x (0, 1) + 4
     *   (0, 1) x (-1, ) x (0, 1) + 4
     *   (0, 1) x (0, 1) x (-1, ) + 4 -> N = 20
     *
     * > To simplify, we define all the 27 neighbor indices
     *   (not necessary to assign them all in pre-processing, but it is
     *   anyway not too expensive to do so).
     *   (-1, 0, 1) ^ 3 -> N = 27
     *   The 3D neighbor indices are converted to 1D in LinearizeNeighborIndex.
     *
     * ---
     *
     * Given (x, y, z) in voxel units, the optimized query should be:
     * 0. > Decide the subvolume this kernel is working on (Xsv, can be
     *      stored in @target_subvolume_entry_array_ beforehand),
     *      and pre-allocate and store neighbor @subvolumes in shared memory.
     *
     *      For each voxel (x, y, z):
     * 1. > Get voxel coordinate in this specific volume
     *      Vector3f Xlocal = voxel[f]_global_to_local(x, y, z, Xsv);
     *
     * 2. > Check if it is on boundary (danger zone)
     *      OnBoundary[f](Xlocal)
     *
     * 3. > If not, directly use subvolumes[LinearizeNeighborIndex(0, 0, 0)]
     *      to access/interpolate data
     *
     * 4. > If so, use XXXOnBorderAt(Xlocal) to interpolate.
     *      These functions will first query neighbors for each point to
     *      interpolate, and access them from pre-allocated subvolumes
     *        Vector3i dXsv = NeighborIndexOfBoundaryVoxel(Xlocal)
     *        subvolumes[LinearizeNeighborIndex(dXsv)].tsdf() ...
     */
    __DEVICE__ void QuerySubvolumeWithNeighborIndex(
        const Vector3i &Xsv, int dxsv, int dysv, int dzsv,
        UniformTSDFVolumeCudaServer<N> **subvolume);

    __DEVICE__ inline Vector3i NeighborIndexOfBoundaryVoxel(
        int xlocal, int ylocal, int zlocal);
    __DEVICE__ inline Vector3i NeighborIndexOfBoundaryVoxel(
        const Vector3i &Xlocal);

    __DEVICE__ inline int LinearizeNeighborIndex(int dxsv, int dysv, int dzsv);
    __DEVICE__ inline int LinearizeNeighborIndex(const Vector3i &dXsv);

    /** Note we assume we already know @offset is in @block.
     *  For interpolation, boundary regions are [N-1~N] for float, none for int
     *  For gradient, boundary regions are [N-2 ~ N) and [0 ~ 1) **/
    __DEVICE__ inline bool OnBoundary(
        int xlocal, int ylocal, int zlocal, bool for_gradient = false);
    __DEVICE__ inline bool OnBoundary(
        const Vector3i &Xlocal, bool for_gradient = false);
    __DEVICE__ inline bool OnBoundaryf(
        float xlocal, float ylocal, float zlocal, bool for_gradinet = false);
    __DEVICE__ inline bool OnBoundaryf(
        const Vector3f &Xlocal, bool for_gradient = false);

    /** In these functions range of input indices are [-1, N+1)
     * (xlocal, ylocal, zlocal) is inside subvolumes[IndexOfNeighborSubvolumes(0, 0, 0)]
     **/
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
    __DEVICE__ void Integrate(int xlocal, int ylocal, int zlocal,
                              HashEntry<Vector3i> &target_subvolume_entry,
                              ImageCudaServer<Vector1f> &depth,
                              MonoPinholeCameraCuda &camera,
                              TransformCuda &transform_camera_to_world);
    __DEVICE__ Vector3f RayCasting(int x, int y,
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
                           TransformCuda &transform_volume_to_world);
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
    /** Hash_table based integration is non-trivial,
     *  it requires 3: pre-allocation, get volumes, and integration
     *  NOTE: we cannot merge stage 1 and 2:
     *  - TouchBlocks allocate blocks in parallel.
     *  - If we return only newly allocated volumes, then we fail to capture
     *    already allocated volumes.
     *  - If we capture all the allocated volume indices in parallel, then
     *    there will be duplicates. (thread1 allocate and return, thread2
     *    capture it and return again). **/
    void TouchSubvolumes(ImageCuda<Vector1f> &depth,
                         MonoPinholeCameraCuda &camera,
                         TransformCuda &transform_camera_to_world);
    void GetSubvolumesInFrustum(MonoPinholeCameraCuda &camera,
                                TransformCuda &transform_camera_to_world);
    void IntegrateSubvolumes(ImageCuda<Vector1f> &depth,
                             MonoPinholeCameraCuda &camera,
                             TransformCuda &transform_camera_to_world);

    void Integrate(ImageCuda<Vector1f> &depth,
                   MonoPinholeCameraCuda &camera,
                   TransformCuda &transform_camera_to_world);
    void RayCasting(ImageCuda<Vector3f> &image,
                    MonoPinholeCameraCuda &camera,
                    TransformCuda &transform_camera_to_world);

public:
    SpatialHashTableCuda &hash_table() {
        return hash_table_;
    }
    const SpatialHashTableCuda &hash_table() const {
        return hash_table_;
    }
    ArrayCuda<HashEntry<Vector3i>> &target_subvolume_entry_array() {
        return target_subvolume_entry_array_;
    }
    const ArrayCuda<HashEntry<Vector3i>> &target_subvolume_entry_array() const {
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
void IntegrateSubvolumesKernel(ScalableTSDFVolumeCudaServer<N> server,
                               ImageCudaServer<Vector1f> depth,
                               MonoPinholeCameraCuda camera,
                               TransformCuda transform_camera_to_world);

template<size_t N>
__GLOBAL__
void GetSubvolumesInFrustumKernel(ScalableTSDFVolumeCudaServer<N> server,
                                  MonoPinholeCameraCuda camera,
                                  TransformCuda transform_camera_to_world);

template<size_t N>
__GLOBAL__
void RayCastingKernel(ScalableTSDFVolumeCudaServer<N> server,
                      ImageCudaServer<Vector3f> normal,
                      MonoPinholeCameraCuda camera,
                      TransformCuda transform_camera_to_world);

}
