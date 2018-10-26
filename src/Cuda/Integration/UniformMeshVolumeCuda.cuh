//
// Created by wei on 10/16/18.
//

#pragma once

#include "MarchingCubesConstCuda.h"
#include "UniformMeshVolumeCuda.h"
#include "UniformTSDFVolumeCuda.cuh"
#include <Core/Core.h>

namespace open3d {
/**
 * Server end
 */
template<VertexType type, size_t N>
__device__
void UniformMeshVolumeCudaServer<type, N>::AllocateVertex(
    int x, int y, int z,
    UniformTSDFVolumeCudaServer<N> &tsdf_volume) {

    uchar &table_index = table_indices(x, y, z);
    table_index = 0;

    int tmp_table_index = 0;

    /** There are early returns. #pragma unroll SLOWS it down **/
    for (int i = 0; i < 8; ++i) {
        const int xi = x + shift[i][0];
        const int yi = y + shift[i][1];
        const int zi = z + shift[i][2];

        uchar weight = tsdf_volume.weight(xi, yi, zi);
        if (weight == 0) return;

        float tsdf = tsdf_volume.tsdf(xi, yi, zi);
        if (fabsf(tsdf) > 2 * tsdf_volume.voxel_length_) return;

        tmp_table_index |= ((tsdf < 0) ? (1 << i) : 0);
    }
    if (tmp_table_index == 0 || tmp_table_index == 255) return;
    table_index = (uchar) tmp_table_index;

    /** Tell them they will be extracted. Conflict can be ignored **/
    int edges = edge_table[table_index];
#pragma unroll 12
    for (int i = 0; i < 12; ++i) {
        if (edges & (1 << i)) {
            vertex_indices(x + edge_shift[i][0],
                           y + edge_shift[i][1],
                           z + edge_shift[i][2])(edge_shift[i][3]) =
                VERTEX_TO_ALLOCATE;
        }
    }
}

template<VertexType type, size_t N>
__device__
void UniformMeshVolumeCudaServer<type, N>::ExtractVertex(
    int x, int y, int z,
    UniformTSDFVolumeCudaServer<N> &tsdf_volume) {

    Vector3i &vertex_index = vertex_indices(x, y, z);
    if (vertex_index(0) != VERTEX_TO_ALLOCATE
        && vertex_index(1) != VERTEX_TO_ALLOCATE
        && vertex_index(2) != VERTEX_TO_ALLOCATE)
        return;

    Vector3i X = Vector3i(x, y, z);
    Vector3i offset = Vector3i::Zeros();

    float tsdf_0 = tsdf_volume.tsdf(X);
    Vector3f gradient_0 = tsdf_volume.gradient(X);

#pragma unroll 1
    for (size_t i = 0; i < 3; ++i) {
        if (vertex_index(i) == VERTEX_TO_ALLOCATE) {
            offset(i) = 1;
            Vector3i X_i = X + offset;

            float tsdf_i = tsdf_volume.tsdf(X_i);
            float mu = (0 - tsdf_0) / (tsdf_i - tsdf_0);
            vertex_index(i) = mesh_.vertices().push_back(
                tsdf_volume.voxel_to_world(x + mu * offset(0),
                                           y + mu * offset(1),
                                           z + mu * offset(2)));

            /** Note we share the vertex indices **/
            if (type & VertexWithNormal) {
                mesh_.vertex_normals()[vertex_index(i)] =
                    tsdf_volume.transform_volume_to_world_.Rotate(
                        (1 - mu) * gradient_0 + mu * tsdf_volume.gradient(X_i));
            }

            offset(i) = 0;
        }
    }
}

template<VertexType type, size_t N>
__device__
inline void UniformMeshVolumeCudaServer<type, N>::ExtractTriangle(
    int x, int y, int z) {

    const uchar table_index = table_indices(x, y, z);
    if (table_index == 0 || table_index == 255) return;

    for (int i = 0; i < 16; i += 3) {
        if (tri_table[table_index][i] == -1) return;

        /** Edge index -> neighbor cube index ([0, 1])^3 x vertex index (3) **/
        Vector3i vertex_index;
#pragma unroll 1
        for (int j = 0; j < 3; ++j) {
            /** Edge index **/
            int edge_j = tri_table[table_index][i + j];
            vertex_index(j) = vertex_indices(
                x + edge_shift[edge_j][0],
                y + edge_shift[edge_j][1],
                z + edge_shift[edge_j][2])(edge_shift[edge_j][3]);
        }
        mesh_.triangles().push_back(vertex_index);
    }
}

/**
 * Client end
 */
template<VertexType type, size_t N>
UniformMeshVolumeCuda<type, N>::UniformMeshVolumeCuda() {
    max_vertices_ = -1;
    max_triangles_ = -1;
}

template<VertexType type, size_t N>
UniformMeshVolumeCuda<type, N>::UniformMeshVolumeCuda(
    int max_vertices, int max_triangles) {
    Create(max_vertices, max_triangles);
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
        PrintError("Already created. Stop re-creating!\n");
        return;
    }

    assert(max_vertices > 0 && max_triangles > 0);

    server_ = std::make_shared<UniformMeshVolumeCudaServer<type, N>>();
    max_triangles_ = max_triangles;
    max_vertices_ = max_vertices;

    const int NNN = N * N * N;
    CheckCuda(cudaMalloc(&server_->table_indices_, sizeof(uchar) * NNN));
    CheckCuda(cudaMalloc(&server_->vertex_indices_, sizeof(Vector3i) * NNN));
    mesh_.Create(max_vertices_, max_triangles_);

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

/** Reset only have to be performed once on initialization:
 * 1. table_indices_ will be reset in kernels;
 * 2. None of the vertex_indices_ will be -1 after this reset, because
 *  - The not effected vertex indices will remain 0;
 *  - The effected vertex indices will be >= 0 after being assigned address.
 **/
template<VertexType type, size_t N>
void UniformMeshVolumeCuda<type, N>::Reset() {
    if (server_ != nullptr) {
        const size_t NNN = N * N * N;
        CheckCuda(cudaMemset(server_->table_indices_, 0,
                             sizeof(uchar) * NNN));
        CheckCuda(cudaMemset(server_->vertex_indices_, 0,
                             sizeof(Vector3i) * NNN));
        mesh_.Reset();
    }
}

template<VertexType type, size_t N>
void UniformMeshVolumeCuda<type, N>::UpdateServer() {
    if (server_ != nullptr) {
        server_->mesh_ = *mesh_.server();
    }
}

template<VertexType type, size_t N>
void UniformMeshVolumeCuda<type, N>::VertexAllocation(
    UniformTSDFVolumeCuda<N> &tsdf_volume) {

    Timer timer;
    timer.Start();

    const int num_blocks = DIV_CEILING(N, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    MarchingCubesVertexAllocationKernel << < blocks, threads >> > (
        *server_, *tsdf_volume.server());
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    timer.Stop();
    PrintInfo("Allocation takes %f milliseconds\n", timer.GetDuration());
}

template<VertexType type, size_t N>
void UniformMeshVolumeCuda<type, N>::VertexExtraction(
    UniformTSDFVolumeCuda<N> &tsdf_volume) {

    Timer timer;
    timer.Start();

    const int num_blocks = DIV_CEILING(N, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    MarchingCubesVertexExtractionKernel << < blocks, threads >> > (
        *server_, *tsdf_volume.server());
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    timer.Stop();
    PrintInfo("Extraction takes %f milliseconds\n", timer.GetDuration());
}

template<VertexType type, size_t N>
void UniformMeshVolumeCuda<type, N>::TriangleExtraction() {

    Timer timer;
    timer.Start();

    const int num_blocks = DIV_CEILING(N, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    MarchingCubesTriangleExtractionKernel << < blocks, threads >> > (*server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    timer.Stop();
    PrintInfo("Triangulation takes %f milliseconds\n", timer.GetDuration());
}

template<VertexType type, size_t N>
void UniformMeshVolumeCuda<type, N>::MarchingCubes(
    UniformTSDFVolumeCuda<N> &tsdf_volume) {

    mesh_.Reset();

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