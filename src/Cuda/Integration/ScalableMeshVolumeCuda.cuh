//
// Created by wei on 10/23/18.
//

#pragma once

#include "MarchingCubesConstCuda.h"
#include "ScalableMeshVolumeCuda.h"
#include "ScalableTSDFVolumeCuda.cuh"
#include <Core/Core.h>

namespace open3d {
/**
 * Server end
 */
template<VertexType type, size_t N>
__device__
void ScalableMeshVolumeCudaServer<type, N>::AllocateVertex(
    int xlocal, int ylocal, int zlocal, int subvolume_idx,
    ScalableTSDFVolumeCudaServer<N> &tsdf_volume,
    UniformTSDFVolumeCudaServer<N> **neighbor_subvolumes) {

//    uchar &table_index = table_indices(xlocal, ylocal, zlocal, subvolume_idx);
//    table_index = 0;
//
//    int tmp_table_index = 0;
//
//    /** There are early returns. #pragma unroll SLOWS it down **/
//    for (int i = 0; i < 8; ++i) {
//        const int xi = xlocal + shift[i][0];
//        const int yi = ylocal + shift[i][1];
//        const int zi = zlocal + shift[i][2];
//
//        uchar weight = tsdf_volume.weight(xi, yi, zi);
//        if (weight == 0) return;
//
//        float tsdf = tsdf_volume.tsdf(xi, yi, zi);
//        if (fabsf(tsdf) > 2 * tsdf_volume.voxel_length_) return;
//
//        tmp_table_index |= ((tsdf < 0) ? (1 << i) : 0);
//    }
//    if (tmp_table_index == 0 || tmp_table_index == 255) return;
//    table_index = (uchar) tmp_table_index;
//
//    /** Tell them they will be extracted. Conflict can be ignored **/
//    int edges = edge_table[table_index];
//#pragma unroll 12
//    for (int i = 0; i < 12; ++i) {
//        if (edges & (1 << i)) {
//            vertex_indices(x + edge_shift[i][0],
//                           y + edge_shift[i][1],
//                           z + edge_shift[i][2])(edge_shift[i][3]) =
//                VERTEX_TO_ALLOCATE;
//        }
//    }
}

/**
 * Client end
 */
template<VertexType type, size_t N>
ScalableMeshVolumeCuda<type, N>::ScalableMeshVolumeCuda() {
    max_subvolumes_ = -1;
    max_vertices_ = -1;
    max_triangles_ = -1;
}

template<VertexType type, size_t N>
ScalableMeshVolumeCuda<type, N>::ScalableMeshVolumeCuda(
    int max_subvolumes, int max_vertices, int max_triangles) {
    Create(max_subvolumes, max_vertices, max_triangles);
}

template<VertexType type, size_t N>
ScalableMeshVolumeCuda<type, N>::ScalableMeshVolumeCuda(
    const ScalableMeshVolumeCuda<type, N> &other) {
    max_subvolumes_ = other.max_subvolumes_;
    max_vertices_ = other.max_vertices_;
    max_triangles_ = other.max_triangles_;

    server_ = other.server();
    mesh_ = other.mesh();
}

template<VertexType type, size_t N>
ScalableMeshVolumeCuda<type, N> &ScalableMeshVolumeCuda<type, N>::operator=(
    const ScalableMeshVolumeCuda<type, N> &other) {
    if (this != &other) {
        max_subvolumes_ = other.max_subvolumes_;
        max_vertices_ = other.max_vertices_;
        max_triangles_ = other.max_triangles_;

        server_ = other.server();
        mesh_ = other.mesh();
    }
    return *this;
}

template<VertexType type, size_t N>
ScalableMeshVolumeCuda<type, N>::~ScalableMeshVolumeCuda() {
    Release();
}

template<VertexType type, size_t N>
void ScalableMeshVolumeCuda<type, N>::Create(
    int max_subvolumes, int max_vertices, int max_triangles) {
    if (server_ != nullptr) {
        PrintError("Already created. Stop re-creating!\n");
        return;
    }

    assert(max_subvolumes > 0 && max_vertices > 0 && max_triangles > 0);

    server_ = std::make_shared<ScalableMeshVolumeCudaServer<type, N>>();
    max_subvolumes_ = max_subvolumes;
    max_vertices_ = max_vertices;
    max_triangles_ = max_triangles;

    const int NNN = N * N * N;
    CheckCuda(cudaMalloc(&server_->table_indices_memory_pool_,
                         sizeof(uchar) * NNN * max_subvolumes_));
    CheckCuda(cudaMalloc(&server_->vertex_indices_memory_pool_,
                         sizeof(Vector3i) * NNN * max_subvolumes_));
    mesh_.Create(max_vertices_, max_triangles_);

    UpdateServer();
    Reset();
}

template<VertexType type, size_t N>
void ScalableMeshVolumeCuda<type, N>::Release() {
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

template<VertexType type, size_t N>
void ScalableMeshVolumeCuda<type, N>::Reset() {
    if (server_ != nullptr) {
        const size_t NNN = N * N * N;
        CheckCuda(cudaMemset(server_->table_indices_memory_pool_, 0,
                             sizeof(uchar) * NNN * max_subvolumes_));
        CheckCuda(cudaMemset(server_->vertex_indices_memory_pool_, 0,
                             sizeof(Vector3i) * NNN * max_subvolumes_));
        mesh_.Reset();
    }
}

template<VertexType type, size_t N>
void ScalableMeshVolumeCuda<type, N>::UpdateServer() {
    if (server_ != nullptr) {
        server_->mesh_ = *mesh_.server();
    }
}

template<VertexType type, size_t N>
void ScalableMeshVolumeCuda<type, N>::VertexAllocation(
    ScalableTSDFVolumeCuda<N> &tsdf_volume) {

    Timer timer;
    timer.Start();

    const dim3 blocks(active_subvolumes_);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    MarchingCubesVertexAllocationKernel << < blocks, threads >> > (
        *server_, *tsdf_volume.server());
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    timer.Stop();
    PrintInfo("Allocation takes %f milliseconds\n", timer.GetDuration());
}

template<VertexType type, size_t N>
void ScalableMeshVolumeCuda<type, N>::VertexExtraction(
    ScalableTSDFVolumeCuda<N> &tsdf_volume) {
    Timer timer;
    timer.Start();

    const dim3 blocks(active_subvolumes_);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    MarchingCubesVertexExtractionKernel << < blocks, threads >> > (
        *server_, *tsdf_volume.server());
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    timer.Stop();
    PrintInfo("Extraction takes %f milliseconds\n", timer.GetDuration());
}

template<VertexType type, size_t N>
void ScalableMeshVolumeCuda<type, N>::TriangleExtraction() {
    Timer timer;
    timer.Start();

    const dim3 blocks(active_subvolumes_);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    MarchingCubesTriangleExtractionKernel << < blocks, threads >> > (*server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    timer.Stop();
    PrintInfo("Triangulation takes %f milliseconds\n", timer.GetDuration());
}

template<VertexType type, size_t N>
void ScalableMeshVolumeCuda<type, N>::MarchingCubes(
    ScalableTSDFVolumeCuda<N> &tsdf_volume) {

    mesh_.Reset();
    active_subvolumes_ = tsdf_volume.active_subvolume_entry_array().size();
    if (active_subvolumes_ <= 0) {
        PrintError("Invalid active subvolumes!\n");
        return;
    }

    VertexAllocation(tsdf_volume);
    VertexExtraction(tsdf_volume);

    TriangleExtraction();

    if (type & VertexWithNormal) {
        mesh_.vertex_normals().set_size(mesh_.vertices().size());
    }
    if (type & VertexWithColor) {
        mesh_.vertex_colors().set_size(mesh_.vertices().size());
    }
}
}