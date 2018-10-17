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
     * We pre-allocate all the data,
     * and ONLY the iterator (index) of @vertices_ is carefully maintained by
     * atomicAdd in array.push_back;
     * @vertex_normals_, @vertex_colros_ just
     * REUSE the iterator (index) of @vertices_
     *
     * > Example: (in a cuda device/global function)
     *   int idx = vertices_.push_back(vertex);
     *   vertex_normals_[idx] = vertex_normal;
     *   vertex_colors_[idx] = vertex_color;
     *
     * Here are some reasons for this choice:
     * 1. We don't want to mess up the iterators by multiple atomicAdds
     *    -- the indices WILL BE inconsistent.
     * 2. We don't want to pack them up in a large struct and use template class
     *    -- the implementation can be even more complex and hard to maintain;
     *    -- the interleaved storage will require non-trivial efforts to
     *       transfer data into TriangleMesh, or OpenGL handles, for rendering.
     **/
    ArrayCudaServer<Vector3f> vertices_;
    ArrayCudaServer<Vector3f> vertex_normals_;
    ArrayCudaServer<Vector3b> vertex_colors_;
    ArrayCudaServer<Vector3i> triangles_;

public:
    int max_vertices_;
    int max_triangles_;

public:
    inline __DEVICE__ ArrayCudaServer<Vector3i>& triangles() {
        return triangles_;
    }
    inline __DEVICE__ ArrayCudaServer<Vector3f>& vertices() {
        return vertices_;
    }
    inline __DEVICE__ ArrayCudaServer<Vector3f>& vertex_normals() {
        return vertex_normals_;
    }
    inline __DEVICE__ ArrayCudaServer<Vector3b>& vertex_colors() {
        return vertex_colors_;
    }

    inline __DEVICE__ int IndexOf(int x, int y, int z) {
        return int(x + y * N  + z * (N * N));
    }
    inline __DEVICE__ uchar &table_indices(int x, int y, int z) {
        return IndexOf(x, y, z);
    }
    inline __DEVICE__ uchar &table_indices(int i) {
        return table_indices_[i];
    }
    inline __DEVICE__ Vector3i &vertex_indices(int x, int y, int z) {
        return IndexOf(x, y, z);
    }
    inline __DEVICE__ Vector3i &vertex_indices(int i) {
        return vertex_indices_[i];
    }

public:
    friend class UniformMeshVolumeCuda<type, N>;
};

template<VertexType type, size_t N>
class UniformMeshVolumeCuda {
private:
    std::shared_ptr<UniformMeshVolumeCudaServer<type, N> > server_ = nullptr;
    ArrayCuda<Vector3f> vertices_;
    ArrayCuda<Vector3f> vertex_normals_;
    ArrayCuda<Vector3b> vertex_colors_;
    ArrayCuda<Vector3i> triangles_;

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
    void UpdateServer();

    TriangleMeshCuda ToTriangleMeshCuda();
    void ToTriangleMeshCuda(TriangleMeshCuda &mesh);

public:
    ArrayCuda<Vector3i> &triangles() {
        return triangles_;
    }
    const ArrayCuda<Vector3i> &triangles() const {
        return triangles_;
    }
    ArrayCuda<Vector3f> &vertices() {
        return vertices_;
    }
    const ArrayCuda<Vector3f> &vertices() const {
        return vertices_;
    }
    ArrayCuda<Vector3f> &vertex_normals() {
        return vertex_normals_;
    }
    const ArrayCuda<Vector3f> &vertex_normals() const {
        return vertex_normals_;
    }
    ArrayCuda<Vector3b> &vertex_colors() {
        return vertex_colors_;
    }
    const ArrayCuda<Vector3b> &vertex_colors() const {
        return vertex_colors_;
    }

    std::shared_ptr<UniformMeshVolumeCudaServer<type, N>> &server() {
        return server_;
    }
    const std::shared_ptr<UniformMeshVolumeCudaServer<type, N>> &server()
    const {
        return server_;
    }
};
}