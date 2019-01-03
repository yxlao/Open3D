//
// Created by wei on 11/9/18.
//

#pragma once

#include "ScalableMeshVolumeCuda.h"
#include "MarchingCubesConstCuda.h"

#include "ScalableTSDFVolumeCuda.h"

#include <Core/Core.h>
#include <cuda_runtime.h>

namespace open3d {
namespace cuda {
/**
 * Client end
 */
template<size_t N>
ScalableMeshVolumeCuda<N>::ScalableMeshVolumeCuda() {
    max_subvolumes_ = -1;
    vertex_type_ = VertexTypeUnknown;
    max_vertices_ = -1;
    max_triangles_ = -1;
}

template<size_t N>
ScalableMeshVolumeCuda<N>::ScalableMeshVolumeCuda(
    int max_subvolumes,
    VertexType type, int max_vertices, int max_triangles) {
    Create(max_subvolumes, type, max_vertices, max_triangles);
}

template<size_t N>
ScalableMeshVolumeCuda<N>::ScalableMeshVolumeCuda(
    const ScalableMeshVolumeCuda<N> &other) {
    max_subvolumes_ = other.max_subvolumes_;

    vertex_type_ = other.vertex_type_;
    max_vertices_ = other.max_vertices_;
    max_triangles_ = other.max_triangles_;

    server_ = other.server();
    mesh_ = other.mesh();
}

template<size_t N>
ScalableMeshVolumeCuda<N> &ScalableMeshVolumeCuda<N>::operator=(
    const ScalableMeshVolumeCuda<N> &other) {
    if (this != &other) {
        max_subvolumes_ = other.max_subvolumes_;

        vertex_type_ = other.vertex_type_;
        max_vertices_ = other.max_vertices_;
        max_triangles_ = other.max_triangles_;

        server_ = other.server();
        mesh_ = other.mesh();
    }
    return *this;
}

template<size_t N>
ScalableMeshVolumeCuda<N>::~ScalableMeshVolumeCuda() {
    Release();
}

template<size_t N>
void ScalableMeshVolumeCuda<N>::Create(
    int max_subvolumes,
    VertexType type, int max_vertices, int max_triangles) {
    if (server_ != nullptr) {
        PrintError("[ScalableMeshVolumeCuda]: Already created, abort!\n");
        return;
    }

    assert(max_subvolumes > 0 && max_vertices > 0 && max_triangles > 0);

    server_ = std::make_shared<ScalableMeshVolumeCudaDevice<N>>();
    max_subvolumes_ = max_subvolumes;

    vertex_type_ = type;
    max_vertices_ = max_vertices;
    max_triangles_ = max_triangles;

    const int NNN = N * N * N;
    CheckCuda(cudaMalloc(&server_->table_indices_memory_pool_,
                         sizeof(uchar) * NNN * max_subvolumes_));
    CheckCuda(cudaMalloc(&server_->vertex_indices_memory_pool_,
                         sizeof(Vector3i) * NNN * max_subvolumes_));
    mesh_.Create(type, max_vertices_, max_triangles_);

    UpdateServer();
    Reset();
}

template<size_t N>
void ScalableMeshVolumeCuda<N>::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        CheckCuda(cudaFree(server_->table_indices_memory_pool_));
        CheckCuda(cudaFree(server_->vertex_indices_memory_pool_));
    }
    mesh_.Release();
    server_ = nullptr;
    max_subvolumes_ = -1;
    max_vertices_ = -1;
    max_triangles_ = -1;
}

template<size_t N>
void ScalableMeshVolumeCuda<N>::Reset() {
    if (server_ != nullptr) {
        const size_t NNN = N * N * N;
        CheckCuda(cudaMemset(server_->table_indices_memory_pool_, 0,
                             sizeof(uchar) * NNN * max_subvolumes_));
        CheckCuda(cudaMemset(server_->vertex_indices_memory_pool_, 0,
                             sizeof(Vector3i) * NNN * max_subvolumes_));
        mesh_.Reset();
    }
}

template<size_t N>
void ScalableMeshVolumeCuda<N>::UpdateServer() {
    if (server_ != nullptr) {
        server_->mesh_ = *mesh_.server();
    }
}

template<size_t N>
void ScalableMeshVolumeCuda<N>::VertexAllocation(
    ScalableTSDFVolumeCuda<N> &tsdf_volume) {
    assert(server_ != nullptr);

    Timer timer;
    timer.Start();

    ScalableMeshVolumeCudaKernelCaller<N>::
    MarchingCubesVertexAllocationKernelCaller(
        *server_, *tsdf_volume.server(), active_subvolumes_);

    timer.Stop();
    PrintDebug("Allocation takes %f milliseconds\n", timer.GetDuration());
}

template<size_t N>
void ScalableMeshVolumeCuda<N>::VertexExtraction(
    ScalableTSDFVolumeCuda<N> &tsdf_volume) {
    assert(server_ != nullptr);

    Timer timer;
    timer.Start();

    ScalableMeshVolumeCudaKernelCaller<N>::
    MarchingCubesVertexExtractionKernelCaller(
        *server_, *tsdf_volume.server(), active_subvolumes_);

    timer.Stop();
    PrintDebug("Extraction takes %f milliseconds\n", timer.GetDuration());
}

template<size_t N>
void ScalableMeshVolumeCuda<N>::TriangleExtraction(
    ScalableTSDFVolumeCuda<N> &tsdf_volume) {
    assert(server_ != nullptr);

    Timer timer;
    timer.Start();

    ScalableMeshVolumeCudaKernelCaller<N>::
    MarchingCubesTriangleExtractionKernelCaller(
        *server_, *tsdf_volume.server(), active_subvolumes_);

    timer.Stop();
    PrintDebug("Triangulation takes %f milliseconds\n", timer.GetDuration());
}

template<size_t N>
void ScalableMeshVolumeCuda<N>::MarchingCubes(
    ScalableTSDFVolumeCuda<N> &tsdf_volume) {
    assert(server_ != nullptr && vertex_type_ != VertexTypeUnknown);

    mesh_.Reset();
    active_subvolumes_ = tsdf_volume.active_subvolume_entry_array().size();
    PrintDebug("Active subvolumes: %d\n", active_subvolumes_);

    if (active_subvolumes_ <= 0) {
        PrintError("Invalid active subvolume numbers: %d !\n",
                   active_subvolumes_);
        return;
    }

    VertexAllocation(tsdf_volume);
    VertexExtraction(tsdf_volume);

    TriangleExtraction(tsdf_volume);

    if (vertex_type_ & VertexWithNormal) {
        mesh_.vertex_normals().set_iterator(mesh_.vertices().size());
    }
    if (vertex_type_ & VertexWithColor) {
        mesh_.vertex_colors().set_iterator(mesh_.vertices().size());
    }
}
} // cuda
} // open3d