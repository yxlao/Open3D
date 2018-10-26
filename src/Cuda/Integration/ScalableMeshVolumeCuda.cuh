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
    UniformTSDFVolumeCudaServer<N> *subvolume) {

    uchar &table_index = table_indices(xlocal, ylocal, zlocal, subvolume_idx);
    table_index = 0;

    int tmp_table_index = 0;

    /** There are early returns. #pragma unroll SLOWS it down **/
    for (int i = 0; i < 8; ++i) {
        const int xi = xlocal + shift[i][0];
        const int yi = ylocal + shift[i][1];
        const int zi = zlocal + shift[i][2];

        uchar weight = subvolume->weight(xi, yi, zi);
        if (weight == 0) return;

        float tsdf = subvolume->tsdf(xi, yi, zi);
        if (fabsf(tsdf) > 2 * subvolume->voxel_length_) return;

        tmp_table_index |= ((tsdf < 0) ? (1 << i) : 0);
    }
    if (tmp_table_index == 0 || tmp_table_index == 255) return;
    table_index = (uchar) tmp_table_index;

    /** Tell them they will be extracted. Conflict can be ignored **/
    int edges = edge_table[table_index];
#pragma unroll 12
    for (int i = 0; i < 12; ++i) {
        if (edges & (1 << i)) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
            assert(xlocal + edge_shift[i][0] < N);
            assert(ylocal + edge_shift[i][1] < N);
            assert(zlocal + edge_shift[i][2] < N);
#endif
            vertex_indices(xlocal + edge_shift[i][0],
                           ylocal + edge_shift[i][1],
                           zlocal + edge_shift[i][2],
                           subvolume_idx)(edge_shift[i][3])
                = VERTEX_TO_ALLOCATE;
        }
    }
}

template<VertexType type, size_t N>
__device__
void ScalableMeshVolumeCudaServer<type, N>::AllocateVertexOnBoundary(
    int xlocal, int ylocal, int zlocal, int subvolume_idx,
    ScalableTSDFVolumeCudaServer<N> &tsdf_volume,
    int *neighbor_subvolume_indices,
    UniformTSDFVolumeCudaServer<N> **neighbor_subvolumes) {

    uchar &table_index = table_indices(xlocal, ylocal, zlocal, subvolume_idx);
    table_index = 0;

    int tmp_table_index = 0;

    /** There are early returns. #pragma unroll SLOWS it down **/
    for (int i = 0; i < 8; ++i) {
        const int xi = xlocal + shift[i][0];
        const int yi = ylocal + shift[i][1];
        const int zi = zlocal + shift[i][2];

        Vector3i dXsv = tsdf_volume.NeighborIndexOfBoundaryVoxel(xi, yi, zi);
        int neighbor_idx = tsdf_volume.LinearizeNeighborIndex(dXsv);
        UniformTSDFVolumeCudaServer<N>
            *subvolume = neighbor_subvolumes[neighbor_idx];

        if (subvolume == nullptr) return;
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(neighbor_subvolume_indices[neighbor_idx] != NULLPTR_CUDA);
#endif

        const int xi_neighbor = xi - int(N) * dXsv(0);
        const int yi_neighbor = yi - int(N) * dXsv(1);
        const int zi_neighbor = zi - int(N) * dXsv(2);
        uchar weight = subvolume->weight(xi_neighbor, yi_neighbor, zi_neighbor);
        if (weight == 0) return;

        float tsdf = subvolume->tsdf(xi_neighbor, yi_neighbor, zi_neighbor);
        if (fabsf(tsdf) > 2 * subvolume->voxel_length_) return;

        tmp_table_index |= ((tsdf < 0) ? (1 << i) : 0);
    }
    if (tmp_table_index == 0 || tmp_table_index == 255) return;
    table_index = (uchar) tmp_table_index;

    int edges = edge_table[table_index];
    for (int i = 0; i < 12; ++i) {
        if (edges & (1 << i)) {
            const int xi = xlocal + edge_shift[i][0];
            const int yi = ylocal + edge_shift[i][1];
            const int zi = zlocal + edge_shift[i][2];

            Vector3i dXsv = tsdf_volume.NeighborIndexOfBoundaryVoxel(xi, yi, zi);
            int neighbor_idx = tsdf_volume.LinearizeNeighborIndex(dXsv);
            int neighbor_subvolume_idx = neighbor_subvolume_indices[neighbor_idx];

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
            assert(neighbor_subvolume_idx != NULLPTR_CUDA);
#endif

            vertex_indices(xi - int(N) * dXsv(0),
                           yi - int(N) * dXsv(1),
                           zi - int(N) * dXsv(2),
                           neighbor_subvolume_idx)(edge_shift[i][3]) =
                VERTEX_TO_ALLOCATE;
        }
    }
}

template<VertexType type, size_t N>
__device__
void ScalableMeshVolumeCudaServer<type, N>::ExtractVertex(
    int xlocal, int ylocal, int zlocal,
    int subvolume_idx, const Vector3i &Xsv,
    ScalableTSDFVolumeCudaServer<N> &tsdf_volume,
    UniformTSDFVolumeCudaServer<N> *subvolume) {

    Vector3i
        &vertex_index = vertex_indices(xlocal, ylocal, zlocal, subvolume_idx);
    if (vertex_index(0) != VERTEX_TO_ALLOCATE
        && vertex_index(1) != VERTEX_TO_ALLOCATE
        && vertex_index(2) != VERTEX_TO_ALLOCATE)
        return;

    Vector3i Xlocal = Vector3i(xlocal, ylocal, zlocal);
    Vector3i offset = Vector3i::Zeros();

    float tsdf_0 = subvolume->tsdf(Xlocal);
    Vector3f gradient_0 = subvolume->gradient(Xlocal);

#pragma unroll 1
    for (size_t i = 0; i < 3; ++i) {
        if (vertex_index(i) == VERTEX_TO_ALLOCATE) {
            offset(i) = 1;
            Vector3i Xlocal_i = Xlocal + offset;

            float tsdf_i = subvolume->tsdf(Xlocal_i);
            float mu = (0 - tsdf_0) / (tsdf_i - tsdf_0);

            Vector3f X_i = tsdf_volume.voxelf_local_to_global(
                Xlocal(0) + mu * offset(0),
                Xlocal(1) + mu * offset(1),
                Xlocal(2) + mu * offset(2),
                Xsv);

            vertex_index(i) = mesh_.vertices().push_back(
                tsdf_volume.voxel_to_world(X_i));

            /** Note we share the vertex indices **/
            if (type & VertexWithNormal) {
                mesh_.vertex_normals()[vertex_index(i)] =
                    tsdf_volume.transform_volume_to_world_.Rotate(
                        (1 - mu) * gradient_0 + mu * subvolume->gradient(Xlocal_i));
            }

            offset(i) = 0;
        }
    }
}

template<VertexType type, size_t N>
__device__
void ScalableMeshVolumeCudaServer<type, N>::ExtractVertexOnBoundary(
    int xlocal, int ylocal, int zlocal,
    int subvolume_idx, const Vector3i &Xsv,
    ScalableTSDFVolumeCudaServer<N> &tsdf_volume,
    int *neighbor_subvolume_indices,
    UniformTSDFVolumeCudaServer<N> **neighbor_subvolumes) {

    Vector3i
        &vertex_index = vertex_indices(xlocal, ylocal, zlocal, subvolume_idx);
    if (vertex_index(0) != VERTEX_TO_ALLOCATE
        && vertex_index(1) != VERTEX_TO_ALLOCATE
        && vertex_index(2) != VERTEX_TO_ALLOCATE)
        return;

    Vector3i Xlocal = Vector3i(xlocal, ylocal, zlocal);
    Vector3i offset = Vector3i::Zeros();

    float tsdf_0 = neighbor_subvolumes[13]->tsdf(Xlocal);

    Vector3f gradient_0 = tsdf_volume.gradient(Xlocal, neighbor_subvolumes);

#pragma unroll 1
    for (size_t i = 0; i < 3; ++i) {
        if (vertex_index(i) == VERTEX_TO_ALLOCATE) {
            offset(i) = 1;
            Vector3i Xlocal_i = Xlocal + offset;
            Vector3i dXsv = tsdf_volume.NeighborIndexOfBoundaryVoxel(Xlocal_i);
            int k = tsdf_volume.LinearizeNeighborIndex(dXsv);

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
            assert(neighbor_subvolumes[k] != nullptr);
#endif

            float tsdf_i = neighbor_subvolumes[k]->tsdf(
                Xlocal_i(0) - dXsv(0) * int(N),
                Xlocal_i(1) - dXsv(1) * int(N),
                Xlocal_i(2) - dXsv(2) * int(N));

            float mu = (0 - tsdf_0) / (tsdf_i - tsdf_0);

            Vector3f X_i = tsdf_volume.voxelf_local_to_global(
                Xlocal(0) + mu * offset(0),
                Xlocal(1) + mu * offset(1),
                Xlocal(2) + mu * offset(2),
                Xsv);

            vertex_index(i) = mesh_.vertices().push_back(
                tsdf_volume.voxel_to_world(X_i));

            /** Note we share the vertex indices **/
            if (type & VertexWithNormal) {
                mesh_.vertex_normals()[vertex_index(i)] =
                    tsdf_volume.transform_volume_to_world_.Rotate(
                        (1 - mu) * gradient_0
                            + mu * tsdf_volume.gradient(Xlocal_i,
                                                        neighbor_subvolumes));
            }

            offset(i) = 0;
        }
    }
}

template<VertexType type, size_t N>
__device__
void ScalableMeshVolumeCudaServer<type, N>::ExtractTriangle(
    int xlocal, int ylocal, int zlocal, int subvolume_idx) {

    const uchar table_index = table_indices(xlocal, ylocal, zlocal, subvolume_idx);
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
                xlocal + edge_shift[edge_j][0],
                ylocal + edge_shift[edge_j][1],
                zlocal + edge_shift[edge_j][2],
                subvolume_idx)(edge_shift[edge_j][3]);
        }
        mesh_.triangles().push_back(vertex_index);
    }
}

template<VertexType type, size_t N>
__device__
void ScalableMeshVolumeCudaServer<type, N>::ExtractTriangleOnBoundary(
    int xlocal, int ylocal, int zlocal, int subvolume_idx,
    ScalableTSDFVolumeCudaServer<N> &tsdf_volume,
    int *neighbor_subvolume_indices) {

    const uchar table_index = table_indices(xlocal, ylocal, zlocal, subvolume_idx);
    if (table_index == 0 || table_index == 255) return;

    for (int i = 0; i < 16; i += 3) {
        if (tri_table[table_index][i] == -1) return;

        /** Edge index -> neighbor cube index ([0, 1])^3 x vertex index (3) **/
        Vector3i vertex_index;
#pragma unroll 1
        for (int j = 0; j < 3; ++j) {
            /** Edge index **/
            int edge_j = tri_table[table_index][i + j];

            Vector3i Xlocal_j = Vector3i(xlocal + edge_shift[edge_j][0],
                                         ylocal + edge_shift[edge_j][1],
                                         zlocal + edge_shift[edge_j][2]);
            Vector3i dXsv = tsdf_volume.NeighborIndexOfBoundaryVoxel(Xlocal_j);
            int k = tsdf_volume.LinearizeNeighborIndex(dXsv);
            int neighbor_subvolume_idx = neighbor_subvolume_indices[k];

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
            assert(neighbor_subvolume_idx != NULLPTR_CUDA);
#endif
            vertex_index(j) = vertex_indices(
                Xlocal_j(0) - int(N) * dXsv(0),
                Xlocal_j(1) - int(N) * dXsv(1),
                Xlocal_j(2) - int(N) * dXsv(2),
                neighbor_subvolume_idx)(edge_shift[edge_j][3]);
        }
        mesh_.triangles().push_back(vertex_index);
    }
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
void ScalableMeshVolumeCuda<type, N>::TriangleExtraction(
    ScalableTSDFVolumeCuda<N> &tsdf_volume) {
    Timer timer;
    timer.Start();

    const dim3 blocks(active_subvolumes_);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    MarchingCubesTriangleExtractionKernel << < blocks, threads >> > (
        *server_, *tsdf_volume.server());
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

    TriangleExtraction(tsdf_volume);

    if (type & VertexWithNormal) {
        mesh_.vertex_normals().set_size(mesh_.vertices().size());
    }
    if (type & VertexWithColor) {
        mesh_.vertex_colors().set_size(mesh_.vertices().size());
    }
}
}