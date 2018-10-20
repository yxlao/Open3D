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
                           z + edge_shift[i][2])(edge_shift[i][3]) = 0;
        }
    }
}

template<VertexType type, size_t N>
__device__
void UniformMeshVolumeCudaServer<type, N>::ExtractVertex(
    int x, int y, int z,
    UniformTSDFVolumeCudaServer<N> &tsdf_volume) {

    Vector3i &vertex_index = vertex_indices(x, y, z);
    bool vx = vertex_index(0) == 0;
    bool vy = vertex_index(1) == 0;
    bool vz = vertex_index(2) == 0;
    if (!vx && !vy && !vz) return;

    float tsdf0 = tsdf_volume.tsdf(x, y, z);

    if (vx) {
        float tsdfx = tsdf_volume.tsdf(x + 1, y, z);
        float mu = (0 - tsdf0) / (tsdfx - tsdf0);
        vertex_index(0) = mesh_.vertices().push_back(
            tsdf_volume.voxel_to_world(x + mu, y, z));

        /** Note we share the vertex indices **/
        if (type & VertexWithNormal) {
            mesh_.vertex_normals()[vertex_index(0)] =
                tsdf_volume.transform_volume_to_world_.Rotate(
                    (1 - mu) * tsdf_volume.gradient(x, y, z)
                        + mu * tsdf_volume.gradient(x + 1, y, z));
        }
    }

    if (vy) {
        float tsdfy = tsdf_volume.tsdf(x, y + 1, z);
        float mu = (0 - tsdf0) / (tsdfy - tsdf0);
        vertex_index(1) = mesh_.vertices().push_back(
            tsdf_volume.voxel_to_world(x, y + mu, z));

        if (type & VertexWithNormal) {
            mesh_.vertex_normals()[vertex_index(1)] =
                tsdf_volume.transform_volume_to_world_.Rotate(
                    (1 - mu) * tsdf_volume.gradient(x, y, z)
                        + mu * tsdf_volume.gradient(x, y + 1, z));
        }
    }

    if (vz) {
        float tsdfz = tsdf_volume.tsdf(x, y, z + 1);
        float mu = (0 - tsdf0) / (tsdfz - tsdf0);
        vertex_index(2) = mesh_.vertices().push_back(
            tsdf_volume.voxel_to_world(x, y, z + mu));

        if (type & VertexWithNormal) {
            mesh_.vertex_normals()[vertex_index(2)] =
                tsdf_volume.transform_volume_to_world_.Rotate(
                    (1 - mu) * tsdf_volume.gradient(x, y, z)
                        + mu * tsdf_volume.gradient(x, y, z + 1));
        }
    }
}

template<VertexType type, size_t N>
__device__
inline void UniformMeshVolumeCudaServer<type, N>::ExtractTriangle(
    int x, int y, int z) {

    const uchar table_index = table_indices(x, y, z);
    if (table_index == 0 || table_index == 255) return;

#pragma unroll 1
    for (int i = 0; i < 16; i += 3) {
        if (tri_table[table_index][i] == -1) return;

        /** Edge index **/
        int edge0 = tri_table[table_index][i + 0];
        int edge1 = tri_table[table_index][i + 1];
        int edge2 = tri_table[table_index][i + 2];

        /** Edge index -> neighbor cube index ([0, 1])^3 x vertex index (3) **/
        Vector3i vertex_index;
        vertex_index(0) = vertex_indices(
            x + edge_shift[edge0][0],
            y + edge_shift[edge0][1],
            z + edge_shift[edge0][2])(edge_shift[edge0][3]);

        vertex_index(1) = vertex_indices(
            x + edge_shift[edge1][0],
            y + edge_shift[edge1][1],
            z + edge_shift[edge1][2])(edge_shift[edge1][3]);

        vertex_index(2) = vertex_indices(
            x + edge_shift[edge2][0],
            y + edge_shift[edge2][1],
            z + edge_shift[edge2][2])(edge_shift[edge2][3]);

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

template<VertexType type, size_t N>
void UniformMeshVolumeCuda<type, N>::MarchingCubes(
    UniformTSDFVolumeCuda<N> &tsdf_volume) {

    const int num_blocks = DIV_CEILING(N, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);

    Timer timer;

    timer.Start();
    MarchingCubesVertexAllocationKernel << < blocks, threads >> > (
        *server_, *tsdf_volume.server());
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
    timer.Stop();
    PrintInfo("Allocation takes %f milliseconds\n", timer.GetDuration());

    timer.Start();
    MarchingCubesVertexExtractionKernel << < blocks, threads >> > (
        *server_, *tsdf_volume.server());
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
    timer.Stop();
    PrintInfo("Extraction takes %f milliseconds\n", timer.GetDuration());

    timer.Start();
    MarchingCubesTriangleExtractionKernel << < blocks, threads >> > (*server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
    timer.Stop();
    PrintInfo("Triangulation takes %f milliseconds\n", timer.GetDuration());

    if (type & VertexWithNormal) {
        mesh_.vertex_normals().set_size(mesh_.vertices().size());
    }
    if (type & VertexWithColor) {
        mesh_.vertex_colors().set_size(mesh_.vertices().size());
    }
}
}