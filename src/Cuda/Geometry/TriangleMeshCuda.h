//
// Created by wei on 10/10/18.
//

#pragma once
#include "GeometryClasses.h"
#include "VectorCuda.h"
#include <Core/Geometry/TriangleMesh.h>
#include <Cuda/Geometry/TransformCuda.h>
#include <Cuda/Container/ArrayCuda.h>
#include <memory>

namespace open3d {

class TriangleMeshCudaServer {
private:
    ArrayCudaServer<Vector3f> vertices_;
    ArrayCudaServer<Vector3f> vertex_normals_;
    ArrayCudaServer<Vector3b> vertex_colors_;
    ArrayCudaServer<Vector3i> triangles_;

public:
    VertexType type_;
    int max_vertices_;
    int max_triangles_;

public:
    __DEVICE__ inline ArrayCudaServer<Vector3f> &vertices() {
        return vertices_;
    }
    __DEVICE__ inline ArrayCudaServer<Vector3f> &vertex_normals() {
        return vertex_normals_;
    }
    __DEVICE__ inline ArrayCudaServer<Vector3b> &vertex_colors() {
        return vertex_colors_;
    }
    __DEVICE__ inline ArrayCudaServer<Vector3i> &triangles() {
        return triangles_;
    }

public:
    friend class TriangleMeshCuda;
};

class TriangleMeshCuda : public Geometry3D {
private:
    std::shared_ptr<TriangleMeshCudaServer> server_ = nullptr;
    ArrayCuda<Vector3f> vertices_;
    ArrayCuda<Vector3f> vertex_normals_;
    ArrayCuda<Vector3b> vertex_colors_;
    ArrayCuda<Vector3i> triangles_;

public:
    VertexType type_;
    int max_vertices_;
    int max_triangles_;

public:
    TriangleMeshCuda();
    TriangleMeshCuda(VertexType type, int max_vertices, int max_triangles);
    TriangleMeshCuda(const TriangleMeshCuda &other);
    TriangleMeshCuda &operator=(const TriangleMeshCuda &other);
    ~TriangleMeshCuda() override;

    void Reset();
    void UpdateServer();

    void Create(VertexType type, int max_vertices, int max_triangles);
    void Release();

    bool HasVertices() const;
    bool HasTriangles() const;
    bool HasVertexNormals() const;
    bool HasVertexColors() const;

    void Upload(TriangleMesh &mesh);
    std::shared_ptr<TriangleMesh> Download();

public:
    void Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetMaxBound() const override;
    void Transform(const Eigen::Matrix4d &transformation) override;

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

    std::shared_ptr<TriangleMeshCudaServer> &server() {
        return server_;
    }
    const std::shared_ptr<TriangleMeshCudaServer> &server() const {
        return server_;
    }
};

__GLOBAL__
void GetMinBoundKernel(TriangleMeshCudaServer server, Vector3f *min_bound);

__GLOBAL__
void GetMaxBoundKernel(TriangleMeshCudaServer server, Vector3f *max_bound);

__GLOBAL__
void TransformKernel(TriangleMeshCudaServer, TransformCuda transform);
}