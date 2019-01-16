//
// Created by wei on 10/10/18.
//

#pragma once

#include "GeometryClasses.h"
#include <Cuda/Common/LinearAlgebraCuda.h>
#include <Cuda/Common/TransformCuda.h>

#include <Cuda/Container/ArrayCuda.h>

#include <Core/Geometry/TriangleMesh.h>

#include <memory>

namespace open3d {
namespace cuda {
class TriangleMeshCudaDevice {
private:
    ArrayCudaDevice<Vector3f> vertices_;
    ArrayCudaDevice<Vector3f> vertex_normals_;
    ArrayCudaDevice<Vector3f> vertex_colors_;
    ArrayCudaDevice<Vector3i> triangles_;

public:
    VertexType type_;
    int max_vertices_;
    int max_triangles_;

public:
    __DEVICE__ inline ArrayCudaDevice<Vector3f> &vertices() {
        return vertices_;
    }
    __DEVICE__ inline ArrayCudaDevice<Vector3f> &vertex_normals() {
        return vertex_normals_;
    }
    __DEVICE__ inline ArrayCudaDevice<Vector3f> &vertex_colors() {
        return vertex_colors_;
    }
    __DEVICE__ inline ArrayCudaDevice<Vector3i> &triangles() {
        return triangles_;
    }

public:
    friend class TriangleMeshCuda;
};

class TriangleMeshCuda : public Geometry3D {
private:
    std::shared_ptr<TriangleMeshCudaDevice> server_ = nullptr;
    ArrayCuda<Vector3f> vertices_;
    ArrayCuda<Vector3f> vertex_normals_;
    ArrayCuda<Vector3f> vertex_colors_;
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
    ArrayCuda<Vector3f> &vertex_colors() {
        return vertex_colors_;
    }
    const ArrayCuda<Vector3f> &vertex_colors() const {
        return vertex_colors_;
    }
    ArrayCuda<Vector3i> &triangles() {
        return triangles_;
    }
    const ArrayCuda<Vector3i> &triangles() const {
        return triangles_;
    }

    std::shared_ptr<TriangleMeshCudaDevice> &server() {
        return server_;
    }
    const std::shared_ptr<TriangleMeshCudaDevice> &server() const {
        return server_;
    }
};

class TriangleMeshCudaKernelCaller {
public:
    static __HOST__ void GetMinBoundKernelCaller(
        TriangleMeshCudaDevice &server,
        ArrayCudaDevice<Vector3f> &min_bound,
        int num_vertices);

    static __HOST__ void GetMaxBoundKernelCaller(
        TriangleMeshCudaDevice &server,
        ArrayCudaDevice<Vector3f> &max_bound,
        int num_vertices);

    static __HOST__ void TransformKernelCaller(
        TriangleMeshCudaDevice &server,
        TransformCuda &transform,
        int num_vertices);
};

__GLOBAL__
void GetMinBoundKernel(TriangleMeshCudaDevice server,
                       ArrayCudaDevice<Vector3f> min_bound);

__GLOBAL__
void GetMaxBoundKernel(TriangleMeshCudaDevice server,
                       ArrayCudaDevice<Vector3f> max_bound);

__GLOBAL__
void TransformKernel(TriangleMeshCudaDevice, TransformCuda transform);
} // cuda
} // open3d