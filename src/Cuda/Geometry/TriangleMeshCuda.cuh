//
// Created by wei on 10/10/18.
//

#pragma once

#include "TriangleMeshCuda.h"
#include <Cuda/Container/ArrayCuda.cuh>

namespace open3d {
TriangleMeshCuda::TriangleMeshCuda()
    : Geometry3D(Geometry::GeometryType::TriangleMeshCuda) {
    type_ = VertexTypeUnknown;

    max_vertices_ = -1;
    max_triangles_ = -1;
}

TriangleMeshCuda::TriangleMeshCuda(
    VertexType type, int max_vertices, int max_triangles)
    : Geometry3D(Geometry::GeometryType::TriangleMeshCuda) {
    Create(type, max_vertices, max_triangles);
}

TriangleMeshCuda::TriangleMeshCuda(const TriangleMeshCuda &other)
    : Geometry3D(Geometry::GeometryType::TriangleMeshCuda) {
    server_ = other.server();

    vertices_ = other.vertices();
    triangles_ = other.triangles();

    vertex_normals_ = other.vertex_normals();
    vertex_colors_ = other.vertex_colors();

    type_ = other.type_;
    max_vertices_ = other.max_triangles_;
    max_triangles_ = other.max_triangles_;
}

TriangleMeshCuda &TriangleMeshCuda::operator=(const TriangleMeshCuda &other) {
    if (this != &other) {
        server_ = other.server();

        vertices_ = other.vertices();
        triangles_ = other.triangles();

        vertex_normals_ = other.vertex_normals();
        vertex_colors_ = other.vertex_colors();

        type_ = other.type_;
        max_vertices_ = other.max_vertices_;
        max_triangles_ = other.max_triangles_;
    }

    return *this;
}

TriangleMeshCuda::~TriangleMeshCuda() {
    Release();
}

void TriangleMeshCuda::Reset() {
    /** No need to clear data **/
    if (type_ == VertexTypeUnknown) {
        PrintError("Unknown vertex type!\n");
    }

    vertices_.set_size(0);
    triangles_.set_size(0);

    if (type_ & VertexWithNormal) {
        vertex_normals_.set_size(0);
    }
    if (type_ & VertexWithColor) {
        vertex_colors_.set_size(0);
    }
}

void TriangleMeshCuda::Create(
    VertexType type, int max_vertices, int max_triangles) {
    assert(max_vertices > 0 && max_triangles > 0);
    if (server_ != nullptr) {
        PrintError("[TriangleMeshCuda] Already created, abort!\n");
        return;
    }

    if (type == VertexTypeUnknown) {
        PrintError("[TriangleMeshCuda] Unknown vertex type, abort!\n");
        return;
    }

    type_ = type;
    max_vertices_ = max_vertices;
    max_triangles_ = max_triangles;

    vertices_.Create(max_vertices_);
    triangles_.Create(max_triangles_);

    if (type_ & VertexWithNormal) {
        vertex_normals_.Create(max_vertices_);
    }
    if (type_ & VertexWithColor) {
        vertex_colors_.Create(max_vertices_);
    }

    server_ = std::make_shared<TriangleMeshCudaServer>();
    UpdateServer();
}

void TriangleMeshCuda::Release() {
    vertices_.Release();
    vertex_normals_.Release();
    vertex_colors_.Release();
    triangles_.Release();

    server_ = nullptr;
    type_ = VertexTypeUnknown;
    max_vertices_ = -1;
    max_triangles_ = -1;
}

void TriangleMeshCuda::UpdateServer() {
    if (server_ != nullptr) {

        server_->type_ = type_;
        server_->max_vertices_ = max_vertices_;
        server_->max_triangles_ = max_triangles_;

        if (type_ != VertexTypeUnknown) {
            server_->vertices_ = *vertices_.server();
            server_->triangles_ = *triangles_.server();
        }

        if (type_ & VertexWithNormal) {
            server_->vertex_normals_ = *vertex_normals_.server();
        }
        if (type_ & VertexWithColor) {
            server_->vertex_colors_ = *vertex_colors_.server();
        }
    }
}

void TriangleMeshCuda::Upload(TriangleMesh &mesh) {
    if (server_ == nullptr) return;

    std::vector<Vector3f> vertices, vertex_normals;
    std::vector<Vector3f> vertex_colors;

    if (!mesh.HasVertices() || !mesh.HasTriangles()) {
        PrintError("Empty mesh!\n");
        return;
    }

    const size_t N = mesh.vertices_.size();
    vertices.resize(N);
    for (int i = 0; i < N; ++i) {
        vertices[i] = Vector3f(mesh.vertices_[i](0),
                               mesh.vertices_[i](1),
                               mesh.vertices_[i](2));
    }
    vertices_.Upload(vertices);

    const size_t M = mesh.triangles_.size();
    std::vector<Vector3i> triangles;
    triangles.resize(M);
    for (int i = 0; i < M; ++i) {
        triangles[i] = Vector3i(mesh.triangles_[i](0),
                                mesh.triangles_[i](1),
                                mesh.triangles_[i](2));
    }
    triangles_.Upload(triangles);

    if ((type_ & VertexWithNormal) && mesh.HasVertexNormals()) {
        vertex_normals.resize(N);
        for (int i = 0; i < N; ++i) {
            vertex_normals[i] = Vector3f(mesh.vertex_normals_[i](0),
                                         mesh.vertex_normals_[i](1),
                                         mesh.vertex_normals_[i](2));
        }
        vertex_normals_.Upload(vertex_normals);
    }

    if ((type_ & VertexWithColor) && mesh.HasVertexColors()) {
        vertex_colors.resize(N);
        for (int i = 0; i < N; ++i) {
            vertex_colors[i] = Vector3f(mesh.vertex_colors_[i](0),
                                        mesh.vertex_colors_[i](1),
                                        mesh.vertex_colors_[i](2));
        }
        vertex_colors_.Upload(vertex_colors);
    }
}

std::shared_ptr<TriangleMesh> TriangleMeshCuda::Download() {
    std::shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();
    if (server_ == nullptr) return mesh;

    if (!HasVertices() || !HasTriangles()) return mesh;

    std::vector<Vector3f> vertices = vertices_.Download();
    std::vector<Vector3i> triangles = triangles_.Download();

    const size_t N = vertices.size();
    mesh->vertices_.resize(N);
    for (int i = 0; i < N; ++i) {
        mesh->vertices_[i] = Eigen::Vector3d(vertices[i](0),
                                             vertices[i](1),
                                             vertices[i](2));
    }

    const size_t M = triangles.size();
    mesh->triangles_.resize(M);
    for (int i = 0; i < M; ++i) {
        mesh->triangles_[i] = Eigen::Vector3i(triangles[i](0),
                                              triangles[i](1),
                                              triangles[i](2));
    }

    if (HasVertexNormals()) {
        std::vector<Vector3f> vertex_normals = vertex_normals_.Download();
        mesh->vertex_normals_.resize(N);
        for (int i = 0; i < N; ++i) {
            mesh->vertex_normals_[i] = Eigen::Vector3d(vertex_normals[i](0),
                                                       vertex_normals[i](1),
                                                       vertex_normals[i](2));
        }
    }

    if (HasVertexColors()) {
        std::vector<Vector3f> vertex_colors = vertex_colors_.Download();
        mesh->vertex_colors_.resize(N);
        for (int i = 0; i < N; ++i) {
            Eigen::Vector3d colord = Eigen::Vector3d(vertex_colors[i](0),
                                                     vertex_colors[i](1),
                                                     vertex_colors[i](2));
            mesh->vertex_colors_[i] = Eigen::Vector3d(min(colord(0), 1.0),
                                                      min(colord(1), 1.0),
                                                      min(colord(2), 1.0));
        }
    }

    return mesh;
}

bool TriangleMeshCuda::HasVertices() const {
    if (type_ == VertexTypeUnknown || server_ == nullptr) return false;
    return vertices_.size() > 0;
}

bool TriangleMeshCuda::HasTriangles() const {
    if (type_ == VertexTypeUnknown || server_ == nullptr) return false;
    return triangles_.size() > 0;
}

bool TriangleMeshCuda::HasVertexNormals() const {
    if ((type_ & VertexWithNormal) == 0 || server_ == nullptr) return false;
    int vertices_size = vertices_.size();
    return vertices_size > 0 && vertices_size == vertex_normals_.size();
}

bool TriangleMeshCuda::HasVertexColors() const {
    if ((type_ & VertexWithColor) == 0 || server_ == nullptr) return false;
    int vertices_size = vertices_.size();
    return vertices_size > 0 && vertices_size == vertex_colors_.size();
}

void TriangleMeshCuda::Clear() {
    Reset();
}

bool TriangleMeshCuda::IsEmpty() const {
    return !HasVertices();
}

Eigen::Vector3d TriangleMeshCuda::GetMinBound() const {
    if (server_ == nullptr) return Eigen::Vector3d(0, 0, 0);

    const int num_vertices = vertices_.size();
    if (num_vertices == 0) return Eigen::Vector3d(0, 0, 0);

    const dim3 blocks(DIV_CEILING(num_vertices, THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    Vector3f min_bound_upload = Vector3f(1e10f, 1e10f, 1e10f);

    Vector3f *min_bound;
    CheckCuda(cudaMalloc(&min_bound, sizeof(Vector3f)));
    CheckCuda(cudaMemcpy(min_bound, &min_bound_upload,
                         sizeof(Vector3f),
                         cudaMemcpyHostToDevice));

    GetMinBoundKernel << < blocks, threads >> > (*server_, min_bound);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    Vector3f min_bound_download;
    CheckCuda(cudaMemcpy(&min_bound_download, min_bound, sizeof(Vector3f),
                         cudaMemcpyDeviceToHost));
    CheckCuda(cudaFree(min_bound));

    return Eigen::Vector3d(min_bound_download(0),
                           min_bound_download(1),
                           min_bound_download(2));
}

Eigen::Vector3d TriangleMeshCuda::GetMaxBound() const {
    if (server_ == nullptr) return Eigen::Vector3d(10, 10, 10);

    const int num_vertices = vertices_.size();
    if (num_vertices == 0) return Eigen::Vector3d(0, 0, 0);

    const dim3 blocks(DIV_CEILING(num_vertices, THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    Vector3f max_bound_upload = Vector3f(-1e10f, -1e10f, -1e10f);

    Vector3f *max_bound;
    CheckCuda(cudaMalloc(&max_bound, sizeof(Vector3f)));
    CheckCuda(cudaMemcpy(max_bound, &max_bound_upload,
                         sizeof(Vector3f),
                         cudaMemcpyHostToDevice));
    GetMaxBoundKernel << < blocks, threads >> > (*server_, max_bound);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    Vector3f max_bound_download;
    CheckCuda(cudaMemcpy(&max_bound_download, max_bound,
                         sizeof(Vector3f),
                         cudaMemcpyDeviceToHost));
    CheckCuda(cudaFree(max_bound));

    return Eigen::Vector3d(max_bound_download(0),
                           max_bound_download(1),
                           max_bound_download(2));
}

void TriangleMeshCuda::Transform(const Eigen::Matrix4d &transformation) {
    if (server_ == nullptr) return;

    const int num_vertices = vertices_.size();
    if (num_vertices == 0) return;

    TransformCuda transformation_cuda;
    transformation_cuda.FromEigen(transformation);

    const dim3 blocks(DIV_CEILING(num_vertices, THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    TransformKernel << < blocks, threads >> > (*server_, transformation_cuda);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
}