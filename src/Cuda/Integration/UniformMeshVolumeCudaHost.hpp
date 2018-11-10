//
// Created by wei on 11/9/18.
//


#include "UniformMeshVolumeCuda.h"
#include "MarchingCubesConstCuda.h"

#include <cuda_runtime.h>
#include <Core/Core.h>

namespace open3d {
/**
 * Client end
 */
template<size_t N>
UniformMeshVolumeCuda<N>::UniformMeshVolumeCuda() {
    vertex_type_ = VertexTypeUnknown;
    max_vertices_ = -1;
    max_triangles_ = -1;
}

template<size_t N>
UniformMeshVolumeCuda<N>::UniformMeshVolumeCuda(
    VertexType type, int max_vertices, int max_triangles) {
    Create(type, max_vertices, max_triangles);
}

template<size_t N>
UniformMeshVolumeCuda<N>::UniformMeshVolumeCuda(
    const UniformMeshVolumeCuda<N> &other) {
    vertex_type_ = other.vertex_type_;
    max_vertices_ = other.max_vertices_;
    max_triangles_ = other.max_triangles_;

    server_ = other.server();
    mesh_ = other.mesh();
}

template<size_t N>
UniformMeshVolumeCuda<N> &UniformMeshVolumeCuda<N>::operator=(
    const UniformMeshVolumeCuda<N> &other) {
    if (this != &other) {
        vertex_type_ = other.vertex_type_;
        max_vertices_ = other.max_vertices_;
        max_triangles_ = other.max_triangles_;

        server_ = other.server();
        mesh_ = other.mesh();
    }
    return *this;
}

template<size_t N>
UniformMeshVolumeCuda<N>::~UniformMeshVolumeCuda() {
    Release();
}

template<size_t N>
void UniformMeshVolumeCuda<N>::Create(
    VertexType type, int max_vertices, int max_triangles) {
    if (server_ != nullptr) {
        PrintError("[UniformMeshVolumeCuda] Already created, abort!\n");
        return;
    }

    assert(max_vertices > 0 && max_triangles > 0);
    assert(type != VertexTypeUnknown);

    server_ = std::make_shared < UniformMeshVolumeCudaServer < N >> ();

    vertex_type_ = type;
    max_triangles_ = max_triangles;
    max_vertices_ = max_vertices;

    const int NNN = N * N * N;
    CheckCuda(cudaMalloc(&server_->table_indices_, sizeof(uchar) * NNN));
    CheckCuda(cudaMalloc(&server_->vertex_indices_, sizeof(Vector3i) * NNN));
    mesh_.Create(vertex_type_, max_vertices_, max_triangles_);

    UpdateServer();
    Reset();
}

template<size_t N>
void UniformMeshVolumeCuda<N>::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        CheckCuda(cudaFree(server_->table_indices_));
        CheckCuda(cudaFree(server_->vertex_indices_));
    }
    mesh_.Release();
    server_ = nullptr;

    vertex_type_ = VertexTypeUnknown;
    max_vertices_ = -1;
    max_triangles_ = -1;
}

/** Reset only have to be performed once on initialization:
 * 1. table_indices_ will be reset in kernels;
 * 2. None of the vertex_indices_ will be -1 after this reset, because
 *  - The not effected vertex indices will remain 0;
 *  - The effected vertex indices will be >= 0 after being assigned address.
 **/
template<size_t N>
void UniformMeshVolumeCuda<N>::Reset() {
    if (server_ != nullptr) {
        const size_t NNN = N * N * N;
        CheckCuda(cudaMemset(server_->table_indices_, 0,
                             sizeof(uchar) * NNN));
        CheckCuda(cudaMemset(server_->vertex_indices_, 0,
                             sizeof(Vector3i) * NNN));
        mesh_.Reset();
    }
}

template<size_t N>
void UniformMeshVolumeCuda<N>::UpdateServer() {
    if (server_ != nullptr) {
        server_->mesh_ = *mesh_.server();
    }
}

template<size_t N>
void UniformMeshVolumeCuda<N>::VertexAllocation(
    UniformTSDFVolumeCuda<N> &tsdf_volume) {
    assert(server_ != nullptr);

    Timer timer;
    timer.Start();

    UniformMeshVolumeCudaKernelCaller<N>::
    MarchingCubesVertexAllocationKernelCaller(
        *server_, *tsdf_volume.server());

    timer.Stop();
    PrintInfo("Allocation takes %f milliseconds\n", timer.GetDuration());
}

template<size_t N>
void UniformMeshVolumeCuda<N>::VertexExtraction(
    UniformTSDFVolumeCuda<N> &tsdf_volume) {
    assert(server_ != nullptr);

    Timer timer;
    timer.Start();

    UniformMeshVolumeCudaKernelCaller<N>::
    MarchingCubesVertexExtractionKernelCaller(
        *server_, *tsdf_volume.server());
    timer.Stop();
    PrintInfo("Extraction takes %f milliseconds\n", timer.GetDuration());
}

template<size_t N>
void UniformMeshVolumeCuda<N>::TriangleExtraction() {
    assert(server_ != nullptr);

    Timer timer;
    timer.Start();

    UniformMeshVolumeCudaKernelCaller<N>::
    MarchingCubesTriangleExtractionKernelCaller(*server_);

    timer.Stop();
    PrintInfo("Triangulation takes %f milliseconds\n", timer.GetDuration());
}

template<size_t N>
void UniformMeshVolumeCuda<N>::MarchingCubes(
    UniformTSDFVolumeCuda<N> &tsdf_volume) {
    assert(server_ != nullptr && vertex_type_ != VertexTypeUnknown);

    mesh_.Reset();

    VertexAllocation(tsdf_volume);
    VertexExtraction(tsdf_volume);

    TriangleExtraction();

    if (vertex_type_ & VertexWithNormal) {
        mesh_.vertex_normals().set_size(mesh_.vertices().size());
    }
    if (vertex_type_ & VertexWithColor) {
        mesh_.vertex_colors().set_size(mesh_.vertices().size());
    }
}
}