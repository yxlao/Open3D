//
// Created by wei on 10/16/18.
//

#pragma once

#include "IntegrationClasses.h"

#include <Geometry/TriangleMesh.h>
#include <Cuda/Common/UtilsCuda.h>
#include <Cuda/Container/ArrayCuda.h>
#include <Cuda/Geometry/VectorCuda.h>
#include <Cuda/Geometry/TriangleMeshCuda.h>

#include <memory>

/** A class that embeds mesh in the volumes **/
/** We need this structure to assure atomic mesh vertex allocation.
 *  Also, we don't want the TSDFVolumes to be too heavy.
 *  When we need to do meshing, we attache this MeshVolume to TSDFVolume **/
namespace open3d {

template<VertexType type, size_t N>
class UniformMeshVolumeCudaServer {
private:
    uchar *table_indices_;
    Vector3i *vertex_indices_;

    /** !!! WARNING !!!:
     * For classes with normals or colors, we pre-allocate all the data,
     * and ONLY the iterator (index) of @vertices_ is carefully maintained by
     * atomicAdd in array.push_back;
     * @vertex_normals_, @vertex_colros_ just
     * REUSE the iterator (index) of @vertices_
     *
     * > Example: (in a cuda device/global function)
     *   int idx = mesh_.vertices().push_back(vertex);
     *   mesh_.vertex_normals()[idx] = vertex_normal;
     *   mesh_.vertex_colors()[idx] = vertex_color;
     *
     * Here are some reasons for this choice:
     * 1. We don't want to mess up the iterators by multiple atomicAdds
     *    -- the indices WILL BE inconsistent.
     * 2. We don't want to pack them up in a large struct and use template class
     *    -- the implementation can be even more complex and hard to maintain;
     *    -- the interleaved storage will require non-trivial efforts to
     *       transfer data into TriangleMesh, or OpenGL handles, for rendering.
     **/
    TriangleMeshCudaServer<type> mesh_;

public:
    int max_vertices_;
    int max_triangles_;

public:
    inline __DEVICE__ int IndexOf(int x, int y, int z) {
        return int(x + y * N  + z * (N * N));
    }
    inline __DEVICE__ uchar &table_indices(int x, int y, int z) {
        return table_indices_[IndexOf(x, y, z)];
    }
    inline __DEVICE__ uchar &table_indices(int i) {
        return table_indices_[i];
    }
    inline __DEVICE__ Vector3i &vertex_indices(int x, int y, int z) {
        return vertex_indices_[IndexOf(x, y, z)];
    }
    inline __DEVICE__ Vector3i &vertex_indices(int i) {
        return vertex_indices_[i];
    }
    inline __DEVICE__ TriangleMeshCudaServer<type> mesh() {
        return mesh_;
    }

public:
    friend class UniformMeshVolumeCuda<type, N>;
};

template<VertexType type, size_t N>
class UniformMeshVolumeCuda {
private:
    std::shared_ptr<UniformMeshVolumeCudaServer<type, N> > server_ = nullptr;
    TriangleMeshCuda<type> mesh_;

public:
    int max_vertices_;
    int max_triangles_;

public:
    UniformMeshVolumeCuda();
    UniformMeshVolumeCuda(const UniformMeshVolumeCuda<type, N> &other);
    UniformMeshVolumeCuda<type, N>& operator=(const
        UniformMeshVolumeCuda<type, N> &other);
    ~UniformMeshVolumeCuda();

    void Create(int max_vertices, int max_triangles);
    void Release();
    void Reset();
    void UpdateServer();

public:
    std::shared_ptr<UniformMeshVolumeCudaServer<type, N>> &server() {
        return server_;
    }
    const std::shared_ptr<UniformMeshVolumeCudaServer<type, N>> &server()
    const {
        return server_;
    }

    TriangleMeshCuda<type> &mesh() {
        return mesh_;
    }
    const TriangleMeshCuda<type> &mesh() const {
        return mesh_;
    }
};
}