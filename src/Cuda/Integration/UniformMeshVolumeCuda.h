//
// Created by wei on 10/16/18.
//

#pragma once

#include "IntegrationClasses.h"
#include "UniformTSDFVolumeCuda.h"

#include <Cuda/Common/UtilsCuda.h>

#include <Cuda/Container/ArrayCuda.h>

#include <Cuda/Common/LinearAlgebraCuda.h>
#include <Cuda/Geometry/TriangleMeshCuda.h>

#include <memory>

/** A class that embeds mesh in the volumes **/
/** We need this structure to assure atomic mesh vertex allocation.
 *  Also, we don't want the TSDFVolumes to be too heavy.
 *  When we need to do meshing, we attache this MeshVolume to TSDFVolume **/
namespace open3d {
namespace cuda {
static const int VERTEX_TO_ALLOCATE = -1;


class UniformMeshVolumeCudaDevice {
private:
    uchar *table_indices_;
    Vector3i *vertex_indices_;

    /** !!! WARNING !!!:
     * For classes with normals or colors, we pre-allocate all the data,
     * and ONLY the @iterator (index) of @vertices_ is carefully maintained by
     * atomicAdd in array.push_back;
     * @vertex_normals_, @vertex_colros_ just
     * REUSE the iterator (index) of @vertices_
     *
     * > Example: (in a cuda device/global function)
     *   int addr = mesh_.vertices().push_back(vertex);
     *   mesh_.vertex_normals()[addr] = vertex_normal;
     *   mesh_.vertex_colors()[addr] = vertex_color;
     *
     * Here are some reasons for this choice:
     * 1. We don't want to mess up the iterators by multiple atomicAdds
     *    -- the indices WILL BE inconsistent.
     * 2. We don't want to pack them up in a large struct and use template class
     *    -- the implementation can be even more complex and hard to maintain;
     *    -- the interleaved storage will require non-trivial efforts to
     *       transfer data into TriangleMesh, or OpenGL handles, for rendering.
     **/
    TriangleMeshCudaDevice mesh_;

public:
    int N_;

    __DEVICE__ inline Vector3i Vectorize(size_t index) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(index < N * N * N);
#endif
        Vector3i ret(0);
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
    __DEVICE__ inline uchar &table_indices(const Vector3i &X) {
        return table_indices_[IndexOf(X)];
    }
    __DEVICE__ inline Vector3i &vertex_indices(const Vector3i &X) {
        return vertex_indices_[IndexOf(X)];
    }
    __DEVICE__ inline TriangleMeshCudaDevice mesh() {
        return mesh_;
    }

public:
    __DEVICE__ void AllocateVertex(
        const Vector3i &X, UniformTSDFVolumeCudaDevice &tsdf_volume);
    __DEVICE__ void ExtractVertex(
        const Vector3i &X, UniformTSDFVolumeCudaDevice &tsdf_volume);
    __DEVICE__ void ExtractTriangle(const Vector3i &X);

public:
    friend class UniformMeshVolumeCuda;
};


class UniformMeshVolumeCuda {
public:
    std::shared_ptr<UniformMeshVolumeCudaDevice > device_ = nullptr;
    TriangleMeshCuda mesh_;

public:
    int N_;
    VertexType vertex_type_;
    int max_vertices_;
    int max_triangles_;

public:
    UniformMeshVolumeCuda();
    UniformMeshVolumeCuda(VertexType type, int N,
                          int max_vertices, int max_triangles);
    UniformMeshVolumeCuda(const UniformMeshVolumeCuda &other);
    UniformMeshVolumeCuda &operator=(
        const UniformMeshVolumeCuda &other);
    ~UniformMeshVolumeCuda();

    void Create(VertexType type, int N, int max_vertices, int max_triangles);
    void Release();
    void Reset();
    void UpdateDevice();

public:
    void VertexAllocation(UniformTSDFVolumeCuda &tsdf_volume);
    void VertexExtraction(UniformTSDFVolumeCuda &tsdf_volume);
    void TriangleExtraction();

    void MarchingCubes(UniformTSDFVolumeCuda &tsdf_volume);

public:
    TriangleMeshCuda &mesh() {
        return mesh_;
    }
    const TriangleMeshCuda &mesh() const {
        return mesh_;
    }
};


class UniformMeshVolumeCudaKernelCaller {
public:
    static void VertexAllocation(
        UniformMeshVolumeCuda &mesher,
        UniformTSDFVolumeCuda &tsdf_volume);

    static __HOST__ void VertexExtraction(
        UniformMeshVolumeCuda &mesher,
        UniformTSDFVolumeCuda &tsdf_volume);

    static __HOST__ void TriangleExtraction(
        UniformMeshVolumeCuda &mesher);
};


__GLOBAL__
void VertexAllocationKernel(
    UniformMeshVolumeCudaDevice server,
    UniformTSDFVolumeCudaDevice tsdf_volume);


__GLOBAL__
void VertexExtractionKernel(
    UniformMeshVolumeCudaDevice server,
    UniformTSDFVolumeCudaDevice tsdf_volume);


__GLOBAL__
void TriangleExtractionKernel(
    UniformMeshVolumeCudaDevice server);
} // cuda
} // open3d