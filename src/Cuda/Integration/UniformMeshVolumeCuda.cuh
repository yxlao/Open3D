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
    mesh_ = other.mesh();
}

template<VertexType type, size_t N>
UniformMeshVolumeCuda<type, N> &UniformMeshVolumeCuda<type, N>::operator=(
    const UniformMeshVolumeCuda<type, N> &other) {
    if (this != &other) {
        max_vertices_ = other.max_vertices_;
        max_triangles_ = other.max_triangles_;

        server_ = other.server();
        mesh_ = other.mesh();
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
    CheckCuda(cudaMalloc(&server_->table_indices_, sizeof(uchar) * NNN));
    CheckCuda(cudaMalloc(&server_->vertex_indices_, sizeof(Vector3i) * NNN));
    mesh_.Create(max_vertices, max_triangles);

    UpdateServer();
    Reset();
}

template<VertexType type, size_t N>
void UniformMeshVolumeCuda<type, N>::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        CheckCuda(cudaFree(server_->table_indices_));
        CheckCuda(cudaFree(server_->vertex_indices_));
    }
    mesh_.Release();
    server_ = nullptr;
    max_vertices_ = -1;
    max_triangles_ = -1;
}

template<VertexType type, size_t N>
void UniformMeshVolumeCuda<type, N>::Reset() {
    if (server_ != nullptr) {
        const size_t NNN = N * N * N;
        CheckCuda(cudaMemset(server_->table_indices_, 0,
                             sizeof(uchar) * NNN));
        CheckCuda(cudaMemset(server_->vertex_indices_, 0xff,
                             sizeof(Vector3i) * NNN));
        mesh_.Reset();
    }
}

template<VertexType type, size_t N>
void UniformMeshVolumeCuda<type, N>::UpdateServer() {
    server_->max_vertices_ = max_vertices_;
    server_->max_triangles_ = max_triangles_;

    server_->mesh_ = *mesh_.server();
}

}