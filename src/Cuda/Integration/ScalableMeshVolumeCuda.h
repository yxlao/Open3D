//
// Created by wei on 10/23/18.
//

#pragma once

#include "IntegrationClasses.h"
#include "UniformTSDFVolumeCuda.h"

#include <Cuda/Common/UtilsCuda.h>

#include <Cuda/Container/ArrayCuda.h>
#include <Cuda/Geometry/TriangleMeshCuda.h>
#include <Cuda/Common/LinearAlgebraCuda.h>

#include <memory>

namespace open3d {
namespace cuda {
/** Almost all the important functions have to be re-written,
 *  so we choose not to reuse UniformMeshVolumeCudaDevice.
 */
class ScalableMeshVolumeCudaDevice {
private:
    uchar *table_indices_memory_pool_;
    Vector3i *vertex_indices_memory_pool_;

    /** Refer to UniformMeshVolumeCudaDevice to check how do we manage
      * vertex indices **/
    TriangleMeshCudaDevice mesh_;

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

    __DEVICE__ inline uchar &table_indices(
        const Vector3i &Xlocal, int subvolume_idx) {
        return table_indices_memory_pool_[IndexOf(Xlocal, subvolume_idx)];
    }

    __DEVICE__ inline Vector3i &vertex_indices(
        const Vector3i &Xlocal, int subvolume_idx) {
        return vertex_indices_memory_pool_[IndexOf(Xlocal, subvolume_idx)];
    }

    __DEVICE__ inline TriangleMeshCudaDevice &mesh() {
        return mesh_;
    }

public:
    /** Same functions as in ScalableTSDFVolumeCuda.
     * Put them here to save calling stack **/
    __DEVICE__ inline Vector3i NeighborOffsetOfBoundaryVoxel(
        const Vector3i &Xlocal);
    __DEVICE__ inline int LinearizeNeighborOffset(const Vector3i &dXsv);
    __DEVICE__ inline Vector3i BoundaryVoxelInNeighbor(
        const Vector3i &Xlocal, const Vector3i &dXsv);

public:
    __DEVICE__ void AllocateVertex(
        const Vector3i &Xlocal, int subvolume_idx,
        UniformTSDFVolumeCudaDevice *subvolume);

    __DEVICE__ void AllocateVertexOnBoundary(
        const Vector3i &Xlocal, int subvolume_idx,
        int *cached_subvolume_indices,
        UniformTSDFVolumeCudaDevice **cached_subvolumes);

    __DEVICE__ void ExtractVertex(
        const Vector3i &Xlocal, int subvolume_idx,
        const Vector3i &Xsv,
        ScalableTSDFVolumeCudaDevice &tsdf_volume,
        UniformTSDFVolumeCudaDevice *subvolume);

    __DEVICE__ void ExtractVertexOnBoundary(
        const Vector3i &Xlocal, int subvolume_idx,
        const Vector3i &Xsv,
        ScalableTSDFVolumeCudaDevice &tsdf_volume,
        UniformTSDFVolumeCudaDevice **cached_subvolumes);

    __DEVICE__ void ExtractTriangle(const Vector3i &Xlocal, int subvolume_idx);

    __DEVICE__ void ExtractTriangleOnBoundary(
        const Vector3i &Xlocal, int subvolume_idx,
        int *cached_subvolume_indices);

public:
    friend class ScalableMeshVolumeCuda;
};

class ScalableMeshVolumeCuda {
public:
    std::shared_ptr<ScalableMeshVolumeCudaDevice> device_ = nullptr;
    TriangleMeshCuda mesh_;

public:
    int N_;

    int active_subvolumes_;

    int max_subvolumes_;

    VertexType vertex_type_;
    int max_vertices_;
    int max_triangles_;

public:
    ScalableMeshVolumeCuda();
    ScalableMeshVolumeCuda(VertexType type,
                           int N, int max_subvolumes,
                           int max_vertices = 2000000,
                           int max_triangles = 4000000);
    ScalableMeshVolumeCuda(const ScalableMeshVolumeCuda &other);
    ScalableMeshVolumeCuda &operator=(const ScalableMeshVolumeCuda &other);
    ~ScalableMeshVolumeCuda();

    void Create(VertexType type, int N, int max_subvolumes,
                int max_vertices, int max_triangles);
    void Release();
    void Reset();
    void UpdateDevice();

public:
    void VertexAllocation(ScalableTSDFVolumeCuda &tsdf_volume);
    void VertexExtraction(ScalableTSDFVolumeCuda &tsdf_volume);
    void TriangleExtraction(ScalableTSDFVolumeCuda &tsdf_volume);

    void MarchingCubes(ScalableTSDFVolumeCuda &tsdf_volume);

public:
    TriangleMeshCuda &mesh() {
        return mesh_;
    }
    const TriangleMeshCuda &mesh() const {
        return mesh_;
    }
};

class ScalableMeshVolumeCudaKernelCaller {
public:
    static void VertexAllocation(ScalableMeshVolumeCuda &server,
                                 ScalableTSDFVolumeCuda &tsdf_volume);
    static void VertexExtraction(ScalableMeshVolumeCuda &server,
                                 ScalableTSDFVolumeCuda &tsdf_volume);
    static void TriangleExtraction(ScalableMeshVolumeCuda &server,
                                   ScalableTSDFVolumeCuda &tsdf_volume);
};


__GLOBAL__
void VertexAllocationKernel(ScalableMeshVolumeCudaDevice server,
                            ScalableTSDFVolumeCudaDevice tsdf_volume);

__GLOBAL__
void VertexExtractionKernel(ScalableMeshVolumeCudaDevice server,
                            ScalableTSDFVolumeCudaDevice tsdf_volume);

__GLOBAL__
void TriangleExtractionKernel(ScalableMeshVolumeCudaDevice server,
                              ScalableTSDFVolumeCudaDevice tsdf_volume);
} // cuda
} // open3d