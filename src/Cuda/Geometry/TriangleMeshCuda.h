//
// Created by wei on 10/10/18.
//

#pragma once

#include "GeometryClasses.h"
#include <Cuda/Common/LinearAlgebraCuda.h>
#include <Cuda/Common/TransformCuda.h>

#include <Cuda/Container/ArrayCuda.h>

#include <Open3D/Geometry/TriangleMesh.h>

#include <memory>

namespace open3d {
namespace cuda {
class TriangleMeshCudaDevice {
public:
    ArrayCudaDevice<Vector3f> vertices_;
    ArrayCudaDevice<Vector3f> vertex_normals_;
    ArrayCudaDevice<Vector3f> vertex_colors_;
    ArrayCudaDevice<Vector3i> triangles_;

public:
    VertexType type_;
    int max_vertices_;
    int max_triangles_;
};

class TriangleMeshCuda : public geometry::Geometry3D {
public:
    std::shared_ptr<TriangleMeshCudaDevice> device_ = nullptr;
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
    void UpdateDevice();

    void Create(VertexType type, int max_vertices, int max_triangles);
    void Release();

    bool HasVertices() const;
    bool HasTriangles() const;
    bool HasVertexNormals() const;
    bool HasVertexColors() const;

    void Upload(geometry::TriangleMesh &mesh);
    std::shared_ptr<geometry::TriangleMesh> Download();

public:
    void Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetMaxBound() const override;
    void Transform(const Eigen::Matrix4d &transformation) override;
};

class TriangleMeshCudaKernelCaller {
public:
    static void GetMinBound(const TriangleMeshCuda &mesh,
                            ArrayCuda<Vector3f> &min_bound);

    static void GetMaxBound(const TriangleMeshCuda &mesh,
                            ArrayCuda<Vector3f> &max_bound);

    static void Transform(TriangleMeshCuda &mesh,
                          TransformCuda &transform);
};

__GLOBAL__
void GetMinBoundKernel(TriangleMeshCudaDevice mesh,
                       ArrayCudaDevice<Vector3f> min_bound);
__GLOBAL__
void GetMaxBoundKernel(TriangleMeshCudaDevice mesh,
                       ArrayCudaDevice<Vector3f> max_bound);
__GLOBAL__
void TransformKernel(TriangleMeshCudaDevice, TransformCuda transform);

} // cuda
} // open3d