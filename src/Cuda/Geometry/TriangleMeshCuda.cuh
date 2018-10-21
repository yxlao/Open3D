//
// Created by wei on 10/10/18.
//

#pragma once

#include "TriangleMeshCuda.h"
#include <Cuda/Container/ArrayCuda.cuh>

namespace open3d {
template<VertexType type>
TriangleMeshCuda<type>::TriangleMeshCuda() {
    max_vertices_ = -1;
    max_triangles_ = -1;
}

template<VertexType type>
TriangleMeshCuda<type>::TriangleMeshCuda(int max_vertices, int max_triangles) {
    max_vertices_ = max_vertices;
    max_triangles_ = max_triangles;
    Create(max_vertices_, max_triangles_);
}

template<VertexType type>
TriangleMeshCuda<type>::TriangleMeshCuda(const TriangleMeshCuda<type> &other) {
    server_ = other.server();

    vertices_ = other.vertices();
    triangles_ = other.triangles();

    vertex_normals_ = other.vertex_normals();
    vertex_colors_ = other.vertex_colors();

    max_vertices_ = other.max_triangles_;
    max_triangles_ = other.max_triangles_;
}

template<VertexType type>
TriangleMeshCuda<type> &TriangleMeshCuda<type>::operator=(
    const TriangleMeshCuda<type> &other) {
    if (this != &other) {
        server_ = other.server();

        vertices_ = other.vertices();
        triangles_ = other.triangles();

        vertex_normals_ = other.vertex_normals();
        vertex_colors_ = other.vertex_colors();

        max_vertices_ = other.max_vertices_;
        max_triangles_ = other.max_triangles_;
    }

    return *this;
}

template<VertexType type>
TriangleMeshCuda<type>::~TriangleMeshCuda() {
    Release();
}

template<VertexType type>
void TriangleMeshCuda<type>::Reset() {
    /** No need to clear data **/
    vertices_.set_size(0);
    triangles_.set_size(0);

    if (type & VertexWithNormal) {
        vertex_normals_.set_size(0);
    }
    if (type & VertexWithColor) {
        vertex_colors_.set_size(0);
    }
}

template<VertexType type>
void TriangleMeshCuda<type>::Create(int max_vertices, int max_triangles) {
    assert(max_vertices > 0 && max_triangles > 0);
    if (server_ != nullptr) {
        PrintError("Already created, stop re-creating!\n");
        return;
    }

    max_vertices_ = max_vertices;
    max_triangles_ = max_triangles;

    vertices_.Create(max_vertices_);
    triangles_.Create(max_triangles_);

    if (type & VertexWithNormal) {
        vertex_normals_.Create(max_vertices_);
    }
    if (type & VertexWithColor) {
        vertex_colors_.Create(max_vertices_);
    }

    server_ = std::make_shared<TriangleMeshCudaServer<type>>();
    UpdateServer();
}

template<VertexType type>
void TriangleMeshCuda<type>::Release() {
    vertices_.Release();
    vertex_normals_.Release();
    vertex_colors_.Release();
    triangles_.Release();

    server_ = nullptr;
    max_vertices_ = -1;
    max_triangles_ = -1;
}

template<VertexType type>
void TriangleMeshCuda<type>::UpdateServer() {
    if (server_ != nullptr) {
        server_->max_vertices_ = max_vertices_;
        server_->max_triangles_ = max_triangles_;

        server_->vertices_ = *vertices_.server();
        server_->triangles_ = *triangles_.server();

        if (type & VertexWithNormal) {
            server_->vertex_normals_ = *vertex_normals_.server();
        }
        if (type & VertexWithColor) {
            server_->vertex_colors_ = *vertex_colors_.server();
        }
    }
}

template<VertexType type>
void TriangleMeshCuda<type>::Upload(TriangleMesh &mesh) {
    if (server_ == nullptr) return;

    std::vector<Vector3f> vertices, vertex_normals;
    std::vector<Vector3b> vertex_colors;

    if (! mesh.HasVertices() || ! mesh.HasTriangles()) return;

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

    if ((type & VertexWithNormal) && mesh.HasVertexNormals()) {
        vertex_normals.resize(N);
        for (int i = 0; i < N; ++i) {
            vertex_normals[i] = Vector3f(mesh.vertex_normals_[i](0),
                                         mesh.vertex_normals_[i](1),
                                         mesh.vertex_normals_[i](2));
        }
        vertex_normals_.Upload(vertex_normals);
    }

    if ((type & VertexWithColor) && mesh.HasVertexColors()) {
        vertex_colors.resize(N);
        for (int i = 0; i < N; ++i) {
            vertex_colors[i] = Vector3b(mesh.vertex_colors_[i](0),
                                        mesh.vertex_colors_[i](1),
                                        mesh.vertex_colors_[i](2));
        }
        vertex_colors_.Upload(vertex_colors);
    }
}

template<VertexType type>
std::shared_ptr<TriangleMesh> TriangleMeshCuda<type>::Download() {
    std::shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();
    if (server_ == nullptr) return mesh;

    if (! HasVertices() || ! HasTriangles()) return mesh;

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
        std::vector<Vector3b> vertex_colors = vertex_colors_.Download();
        mesh->vertex_colors_.resize(N);
        for (int i = 0; i < N; ++i) {
            mesh->vertex_colors_[i] = Eigen::Vector3d(vertex_colors[i](0),
                                                      vertex_colors[i](1),
                                                      vertex_colors[i](2));
        }
    }

    return mesh;
}

template<VertexType type>
bool TriangleMeshCuda<type>::HasVertices() {
    if (server_ == nullptr) return false;
    return vertices_.size() > 0;
}

template<VertexType type>
bool TriangleMeshCuda<type>::HasTriangles() {
    if (server_ == nullptr) return false;
    return triangles_.size() > 0;
}

template<VertexType type>
bool TriangleMeshCuda<type>::HasVertexNormals() {
    if ((type & VertexWithNormal) == 0
    || server_ == nullptr) return false;
    int vertices_size = vertices_.size();
    return vertices_size > 0 && vertices_size == vertex_normals_.size();
}

template<VertexType type>
bool TriangleMeshCuda<type>::HasVertexColors(){
    if ((type & VertexWithColor) == 0
    || server_ == nullptr) return false;
    int vertices_size = vertices_.size();
    return vertices_size > 0 && vertices_size == vertex_colors_.size();
}
}