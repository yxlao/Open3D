//
// Created by wei on 10/16/18.
//

#pragma once

#include "UniformMeshVolumeCuda.h"
#include <Core/Core.h>

namespace open3d {
template<VertexType type, size_t N>
UniformMeshVolumeCuda<type, N>::UniformMeshVolumeCuda() {
    max_vertices_ = -1;
    max_triangles_ = -1;
}

template<VertexType type, size_t N>
UniformMeshVolumeCuda<type, N>::UniformMeshVolumeCuda(
    const UniformMeshVolumeCuda<type, N> &other) {
    max_vertices_ = other.max_vertices_;
    max_triangles_ = other.max_triangles_;

    server_ = other.server();
    vertices_ = other.vertices();
    vertex_normals_ = other.vertex_normals();
    vertex_colors_ = other.vertex_colors();
    triangles_ = other.triangles();
}

template<VertexType type, size_t N>
UniformMeshVolumeCuda<type, N> &UniformMeshVolumeCuda<type, N>::operator=(
    const UniformMeshVolumeCuda<type, N> &other) {
    if (this != &other) {
        max_vertices_ = other.max_vertices_;
        max_triangles_ = other.max_triangles_;

        server_ = other.server();
        vertices_ = other.vertices();
        vertex_normals_ = other.vertex_normals();
        vertex_colors_ = other.vertex_colors();
        triangles_ = other.triangles();
    }
    return *this;
}

template<VertexType type, size_t N>
UniformMeshVolumeCuda<type, N>::~UniformMeshVolumeCuda() {
    Release();
}

template<VertexType type, size_t N>
void UniformMeshVolumeCuda<type, N>::Create(
    int max_vertices, int max_triangles) {
    if (server_ != nullptr) {
        PrintError("Already Created. Stop re-creating!\n");
        return;
    }

    assert(max_vertices > 0 && max_triangles > 0);

    server_ = std::make_shared<UniformMeshVolumeCudaServer<type, N>>();
    max_triangles_ = max_triangles;
    max_vertices_ = max_vertices;

    const int NNN = N * N * N;
    CheckCuda(cudaMalloc(&server_->table_indices_,
                        sizeof(uchar) * NNN));
    CheckCuda(cudaMalloc(&server_->vertex_indices_,
                         sizeof(Vector3i) * NNN));
    triangles_.Create(max_triangles);
    vertices_.Create(max_vertices);
    if (type == VertexWithNormal || type == VertexWithNormalAndColor) {
        vertex_normals_.Create(max_vertices);
    }
    if (type == VertexWithColor || type == VertexWithNormalAndColor) {
        vertex_colors_.Create(max_vertices);
    }

    UpdateServer();
}

template<VertexType type, size_t N>
void UniformMeshVolumeCuda<type, N>::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        CheckCuda(cudaFree(server_->table_indices_));
        CheckCuda(cudaFree(server_->vertex_indices_));
    }

    vertices_.Release();
    vertex_normals_.Release();
    vertex_colors_.Release();
    triangles_.Release();

    server_ = nullptr;
    max_vertices_ = -1;
    max_triangles_ = -1;
}

template<VertexType type, size_t N>
void UniformMeshVolumeCuda<type, N>::UpdateServer() {
    server_->max_vertices_ = max_vertices_;
    server_->max_triangles_ = max_triangles_;

    server_->vertices_ = *vertices_.server();
    server_->vertex_normals_ = *vertex_normals_.server();
    server_->vertex_colors_ = *vertex_colors_.server();
    server_->triangles_ = *triangles_.server();
}

//template<VertexType type, size_t N>
//TriangleMeshCuda UniformMeshVolumeCuda<type, N>::ToTriangleMeshCuda(){
//    /** No way to copy interleaved data. Must launch a kernel call **/
//    TriangleMeshCuda mesh;
//    ToTriangleMeshCuda(mesh);
//    return mesh;
//}
//
//template<VertexType type, size_t N>
//void UniformMeshVolumeCuda<type, N>::ToTriangleMeshCuda(
//    TriangleMeshCuda &mesh) {
//    mesh.vertices() = vertices_;
//    mesh.vertex_normals() = vertex_normals_;
//    mesh.vertex_colors() = vertex_colors_;
//    mesh.triangles() = triangles_;
//    mesh.max_vertices_ = max_vertices_;
//    mesh.max_triangles_ = max_triangles_;
//
//    mesh.UpdateServer();
//}

}