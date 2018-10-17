//
// Created by wei on 10/10/18.
//

#pragma once
#include "GeometryClasses.h"
#include "VectorCuda.h"
#include <Core/Geometry/TriangleMesh.h>
#include <Cuda/Container/ArrayCuda.h>
#include <memory>

namespace open3d {

template<VertexType type>
class TriangleMeshCudaServer {
private:
    ArrayCudaServer<Vector3f> vertices_;
    ArrayCudaServer<Vector3f> vertex_normals_;
    ArrayCudaServer<Vector3b> vertex_colors_;
    ArrayCudaServer<Vector3i> triangles_;

public:
    int max_vertices_;
    int max_triangles_;

public:
    inline __DEVICE__ ArrayCudaServer<Vector3f>& vertices() {
        return vertices_;
    }
    inline __DEVICE__ ArrayCudaServer<Vector3f>& vertex_normals() {
        return vertex_normals_;
    }
    inline __DEVICE__ ArrayCudaServer<Vector3b> &vertex_colors() {
        return vertex_colors_;
    }
    inline __DEVICE__ ArrayCudaServer<Vector3i> &triangles() {
        return triangles_;
    }

public:
    friend class TriangleMeshCuda<type>;
};

template <VertexType type>
class TriangleMeshCuda {
private:
    std::shared_ptr<TriangleMeshCudaServer<type> > server_ = nullptr;
    ArrayCuda<Vector3f> vertices_;
    ArrayCuda<Vector3f> vertex_normals_;
    ArrayCuda<Vector3b> vertex_colors_;
    ArrayCuda<Vector3i> triangles_;

public:
    int max_vertices_;
    int max_triangles_;

public:
    TriangleMeshCuda();
    TriangleMeshCuda(int max_vertices, int max_triangles);
    TriangleMeshCuda(const TriangleMeshCuda& other);
    TriangleMeshCuda& operator= (const TriangleMeshCuda& other);
    ~TriangleMeshCuda();

    void Reset();
    void UpdateServer();

    void Create(int max_vertices, int max_triangles);
    void Release();

    bool HasVertices();
    bool HasTriangles();
    bool HasVertexNormals();
    bool HasVertexColors();

    void Upload(TriangleMesh &mesh);
    std::shared_ptr<TriangleMesh> Download();

public:
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
    ArrayCuda<Vector3i> &triangles() {
        return triangles_;
    }
    const ArrayCuda<Vector3i> &triangles() const {
        return triangles_;
    }

    std::shared_ptr<TriangleMeshCudaServer<type>>& server() {
        return server_;
    }
    const std::shared_ptr<TriangleMeshCudaServer<type>>& server() const {
        return server_;
    }
};


}