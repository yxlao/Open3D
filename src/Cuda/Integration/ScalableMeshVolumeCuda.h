//
// Created by wei on 10/23/18.
//

#pragma once

#include "IntegrationClasses.h"
#include "UniformTSDFVolumeCuda.h"

#include <Cuda/Common/UtilsCuda.h>

#include <Cuda/Container/ArrayCuda.h>
#include <Cuda/Geometry/TriangleMeshCuda.h>
#include <Cuda/Common/VectorCuda.h>

#include <memory>

namespace open3d {
namespace cuda {
/** Almost all the important functions have to be re-written,
 *  so we choose not to reuse UniformMeshVolumeCudaServer.
 */
template<size_t N>
class ScalableMeshVolumeCudaServer {
private:
    uchar *table_indices_memory_pool_;
    Vector3i *vertex_indices_memory_pool_;

    /** Refer to UniformMeshVolumeCudaServer to check how do we manage
      * vertex indices **/
    TriangleMeshCudaServer mesh_;

public:
    __DEVICE__ inline int IndexOf(const Vector3i &Xlocal,
                                  int subvolume_idx) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(Xlocal(0) >= 0 && Xlocal(1) >= 0 && Xlocal(2) >= 0 &&
            subvolume_idx >= 0);
        assert(Xlocal(0) < N && Xlocal(1) < N && Xlocal(2) < N);
#endif
        return int(Xlocal(2) * (N * N) + Xlocal(1) * N + Xlocal(0)
                       + subvolume_idx * (N * N * N));
    }

    __DEVICE__ inline uchar &table_indices(
        const Vector3i &Xlocal, int subvolume_idx) {
        return table_indices_memory_pool_[IndexOf(Xlocal, subvolume_idx)];
    }

    __DEVICE__ inline Vector3i &vertex_indices(
        const Vector3i &Xlocal, int subvolume_idx) {
        return vertex_indices_memory_pool_[IndexOf(Xlocal, subvolume_idx)];
    }

    __DEVICE__ inline TriangleMeshCudaServer &mesh() {
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
        UniformTSDFVolumeCudaServer<N> *subvolume);

    __DEVICE__ void AllocateVertexOnBoundary(
        const Vector3i &Xlocal, int subvolume_idx,
        int *cached_subvolume_indices,
        UniformTSDFVolumeCudaServer<N> **cached_subvolumes);

    __DEVICE__ void ExtractVertex(
        const Vector3i &Xlocal, int subvolume_idx,
        const Vector3i &Xsv,
        ScalableTSDFVolumeCudaServer<N> &tsdf_volume,
        UniformTSDFVolumeCudaServer<N> *subvolume);

    __DEVICE__ void ExtractVertexOnBoundary(
        const Vector3i &Xlocal, int subvolume_idx,
        const Vector3i &Xsv,
        ScalableTSDFVolumeCudaServer<N> &tsdf_volume,
        UniformTSDFVolumeCudaServer<N> **cached_subvolumes);

    __DEVICE__ void ExtractTriangle(const Vector3i &Xlocal, int subvolume_idx);

    __DEVICE__ void ExtractTriangleOnBoundary(
        const Vector3i &Xlocal, int subvolume_idx,
        int *cached_subvolume_indices);

public:
    friend class ScalableMeshVolumeCuda<N>;
};

template<size_t N>
class ScalableMeshVolumeCuda {
private:
    std::shared_ptr<ScalableMeshVolumeCudaServer<N>> server_ = nullptr;
    TriangleMeshCuda mesh_;

public:
    int active_subvolumes_;

    int max_subvolumes_;

    VertexType vertex_type_;
    int max_vertices_;
    int max_triangles_;

public:
    ScalableMeshVolumeCuda();
    ScalableMeshVolumeCuda(int max_subvolumes,
                           VertexType type,
                           int max_vertices,
                           int max_triangles);
    ScalableMeshVolumeCuda(const ScalableMeshVolumeCuda<N> &other);
    ScalableMeshVolumeCuda<N> &operator=(
        const ScalableMeshVolumeCuda<N> &other);
    ~ScalableMeshVolumeCuda();

    void Create(int max_subvolumes,
                VertexType type, int max_vertices, int max_triangles);
    void Release();
    void Reset();
    void UpdateServer();

public:
    void VertexAllocation(ScalableTSDFVolumeCuda<N> &tsdf_volume);
    void VertexExtraction(ScalableTSDFVolumeCuda<N> &tsdf_volume);
    void TriangleExtraction(ScalableTSDFVolumeCuda<N> &tsdf_volume);

    void MarchingCubes(ScalableTSDFVolumeCuda<N> &tsdf_volume);

public:
    std::shared_ptr<ScalableMeshVolumeCudaServer<N>> &server() {
        return server_;
    }
    const std::shared_ptr<ScalableMeshVolumeCudaServer<N>> &server()
    const {
        return server_;
    }

    TriangleMeshCuda &mesh() {
        return mesh_;
    }
    const TriangleMeshCuda &mesh() const {
        return mesh_;
    }
};

template<size_t N>
class ScalableMeshVolumeCudaKernelCaller {
public:
    static __HOST__ void MarchingCubesVertexAllocationKernelCaller(
        ScalableMeshVolumeCudaServer<N> &server,
        ScalableTSDFVolumeCudaServer<N> &tsdf_volume,
        int active_volumes);
    static __HOST__ void MarchingCubesVertexExtractionKernelCaller(
        ScalableMeshVolumeCudaServer<N> &server,
        ScalableTSDFVolumeCudaServer<N> &tsdf_volume,
        int active_volumes);
    static __HOST__ void MarchingCubesTriangleExtractionKernelCaller(
        ScalableMeshVolumeCudaServer<N> &server,
        ScalableTSDFVolumeCudaServer<N> &tsdf_volume,
        int active_volumes);
};

template<size_t N>
__GLOBAL__
void MarchingCubesVertexAllocationKernel(
    ScalableMeshVolumeCudaServer<N> server,
    ScalableTSDFVolumeCudaServer<N> tsdf_volume);

template<size_t N>
__GLOBAL__
void MarchingCubesVertexExtractionKernel(
    ScalableMeshVolumeCudaServer<N> server,
    ScalableTSDFVolumeCudaServer<N> tsdf_volume);

template<size_t N>
__GLOBAL__
void MarchingCubesTriangleExtractionKernel(
    ScalableMeshVolumeCudaServer<N> server,
    ScalableTSDFVolumeCudaServer<N> tsdf_volume);
} // cuda
} // open3d