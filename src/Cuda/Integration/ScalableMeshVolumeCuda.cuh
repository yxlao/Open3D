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
inline Vector3i ScalableMeshVolumeCudaServer<type, N>::
NeighborOffsetOfBoundaryVoxel(const Vector3i &Xlocal) {
    return Vector3i(Xlocal(0) < 0 ? -1 : (Xlocal(0) >= N ? 1 : 0),
                    Xlocal(1) < 0 ? -1 : (Xlocal(1) >= N ? 1 : 0),
                    Xlocal(2) < 0 ? -1 : (Xlocal(2) >= N ? 1 : 0));
}

template<VertexType type, size_t N>
__device__
inline int ScalableMeshVolumeCudaServer<type, N>::LinearizeNeighborOffset(
    const Vector3i &dXsv) {
    /* return (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1); */
    return 9 * dXsv(2) + 3 * dXsv(1) + dXsv(0) + 13;
}

template<VertexType type, size_t N>
__device__
inline Vector3i ScalableMeshVolumeCudaServer<type, N>::BoundaryVoxelInNeighbor(
    const Vector3i &Xlocal, const Vector3i &dXsv) {
    return Vector3i(Xlocal(0) - dXsv(0) * int(N),
                    Xlocal(1) - dXsv(1) * int(N),
                    Xlocal(2) - dXsv(2) * int(N));
}

template<VertexType type, size_t N>
__device__
void ScalableMeshVolumeCudaServer<type, N>::AllocateVertex(
    const Vector3i &Xlocal, int subvolume_idx,
    UniformTSDFVolumeCudaServer<N> *subvolume) {

    uchar &table_index = table_indices(Xlocal, subvolume_idx);
    table_index = 0;

    int tmp_table_index = 0;

    /** There are early returns. #pragma unroll SLOWS it down **/
    for (size_t corner = 0; corner < 8; ++corner) {
        Vector3i Xlocal_corner = Vector3i(Xlocal(0) + shift[corner][0],
                                          Xlocal(1) + shift[corner][1],
                                          Xlocal(2) + shift[corner][2]);

        uchar weight = subvolume->weight(Xlocal_corner);
        if (weight == 0) return;

        float tsdf = subvolume->tsdf(Xlocal_corner);
        if (fabsf(tsdf) > 2 * subvolume->voxel_length_) return;

        tmp_table_index |= ((tsdf < 0) ? (1 << corner) : 0);
    }
    if (tmp_table_index == 0 || tmp_table_index == 255) return;
    table_index = (uchar) tmp_table_index;

    int edges = edge_table[table_index];
#pragma unroll 12
    for (size_t edge = 0; edge < 12; ++edge) {
        if (edges & (1 << edge)) {
            /** Voxel holding the edge **/
            Vector3i Xlocal_edge_holder = Vector3i(
                Xlocal(0) + edge_shift[edge][0],
                Xlocal(1) + edge_shift[edge][1],
                Xlocal(2) + edge_shift[edge][2]);

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
            assert(Xlocal_edge_holder(0) < N
            && Xlocal_edge_holder(1) < N
            && Xlocal_edge_holder(2) < N);
#endif
            vertex_indices(Xlocal_edge_holder, subvolume_idx)
                (edge_shift[edge][3]) = VERTEX_TO_ALLOCATE;
        }
    }
}

template<VertexType type, size_t N>
__device__
void ScalableMeshVolumeCudaServer<type, N>::AllocateVertexOnBoundary(
    const Vector3i &Xlocal, int subvolume_idx,
    int *cached_subvolume_indices,
    UniformTSDFVolumeCudaServer<N> **cached_subvolumes) {

    uchar &table_index = table_indices(Xlocal, subvolume_idx);
    table_index = 0;

    int tmp_table_index = 0;

    /** There are early returns. #pragma unroll SLOWS it down **/
    for (size_t corner = 0; corner < 8; ++corner) {
        Vector3i Xlocal_corner = Vector3i(Xlocal(0) + shift[corner][0],
                                          Xlocal(1) + shift[corner][1],
                                          Xlocal(2) + shift[corner][2]);

        Vector3i dXsv_corner = NeighborOffsetOfBoundaryVoxel(Xlocal_corner);
        int neighbor_idx = LinearizeNeighborOffset(dXsv_corner);
        UniformTSDFVolumeCudaServer<N> *neighbor_subvolume =
            cached_subvolumes[neighbor_idx];

        if (neighbor_subvolume == nullptr) return;
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(cached_subvolume_indices[neighbor_idx] != NULLPTR_CUDA);
#endif

        Vector3i Xlocal_corner_in_neighbor =
            BoundaryVoxelInNeighbor(Xlocal_corner, dXsv_corner);
        uchar weight = neighbor_subvolume->weight(Xlocal_corner_in_neighbor);
        if (weight == 0) return;

        float tsdf = neighbor_subvolume->tsdf(Xlocal_corner_in_neighbor);
        if (fabsf(tsdf) > 2 * neighbor_subvolume->voxel_length_) return;

        tmp_table_index |= ((tsdf < 0) ? (1 << corner) : 0);
    }
    if (tmp_table_index == 0 || tmp_table_index == 255) return;
    table_index = (uchar) tmp_table_index;

    int edges = edge_table[table_index];
    for (size_t edge = 0; edge < 12; ++edge) {
        if (edges & (1 << edge)) {
            /** Voxel holding the edge **/
            Vector3i Xlocal_edge_holder = Vector3i(
                Xlocal(0) + edge_shift[edge][0],
                Xlocal(1) + edge_shift[edge][1],
                Xlocal(2) + edge_shift[edge][2]);

            Vector3i dXsv_edge_holder =
                NeighborOffsetOfBoundaryVoxel(Xlocal_edge_holder);
            int neighbor_idx = LinearizeNeighborOffset(dXsv_edge_holder);
            int neighbor_subvolume_idx = cached_subvolume_indices[neighbor_idx];

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
            assert(neighbor_subvolume_idx != NULLPTR_CUDA);
#endif

            Vector3i Xlocal_edge_holder_in_neighbor = BoundaryVoxelInNeighbor(
                Xlocal_edge_holder, dXsv_edge_holder);

            vertex_indices(
                Xlocal_edge_holder_in_neighbor,neighbor_subvolume_idx)(
                    edge_shift[edge][3]) = VERTEX_TO_ALLOCATE;
        }
    }
}

template<VertexType type, size_t N>
__device__
void ScalableMeshVolumeCudaServer<type, N>::ExtractVertex(
    const Vector3i &Xlocal,
    int subvolume_idx, const Vector3i &Xsv,
    ScalableTSDFVolumeCudaServer<N> &tsdf_volume,
    UniformTSDFVolumeCudaServer<N> *subvolume) {

    Vector3i &voxel_vertex_indices = vertex_indices(Xlocal, subvolume_idx);
    if (voxel_vertex_indices(0) != VERTEX_TO_ALLOCATE
        && voxel_vertex_indices(1) != VERTEX_TO_ALLOCATE
        && voxel_vertex_indices(2) != VERTEX_TO_ALLOCATE)
        return;

    Vector3i axis_offset = Vector3i::Zeros();

    float tsdf_0 = subvolume->tsdf(Xlocal);
    Vector3f gradient_0 = subvolume->gradient(Xlocal);

#pragma unroll 1
    for (size_t axis = 0; axis < 3; ++axis) {
        if (voxel_vertex_indices(axis) == VERTEX_TO_ALLOCATE) {
            axis_offset(axis) = 1;
            Vector3i Xlocal_axis = Xlocal + axis_offset;

            float tsdf_axis = subvolume->tsdf(Xlocal_axis);
            float mu = (0 - tsdf_0) / (tsdf_axis - tsdf_0);

            Vector3f Xlocal_interp_axis = tsdf_volume.voxelf_local_to_global(
                Vector3f(Xlocal(0) + mu * axis_offset(0),
                         Xlocal(1) + mu * axis_offset(1),
                         Xlocal(2) + mu * axis_offset(2)),
                Xsv);

            voxel_vertex_indices(axis) = mesh_.vertices().push_back(
                tsdf_volume.voxelf_to_world(Xlocal_interp_axis));

            if (type & VertexWithNormal) {
                mesh_.vertex_normals()[voxel_vertex_indices(axis)] =
                    tsdf_volume.transform_volume_to_world_.Rotate(
                        (1 - mu) * gradient_0
                            + mu * subvolume->gradient(Xlocal_axis));
            }

            axis_offset(axis) = 0;
        }
    }
}

template<VertexType type, size_t N>
__device__
void ScalableMeshVolumeCudaServer<type, N>::ExtractVertexOnBoundary(
    const Vector3i &Xlocal,
    int subvolume_idx, const Vector3i &Xsv,
    ScalableTSDFVolumeCudaServer<N> &tsdf_volume,
    UniformTSDFVolumeCudaServer<N> **cached_subvolumes) {

    Vector3i &voxel_vertex_indices = vertex_indices(Xlocal, subvolume_idx);
    if (voxel_vertex_indices(0) != VERTEX_TO_ALLOCATE
        && voxel_vertex_indices(1) != VERTEX_TO_ALLOCATE
        && voxel_vertex_indices(2) != VERTEX_TO_ALLOCATE)
        return;

    Vector3i axis_offset = Vector3i::Zeros();

    float tsdf_0 = cached_subvolumes[13]->tsdf(Xlocal);
    Vector3f gradient_0 = tsdf_volume.gradient(Xlocal, cached_subvolumes);

#pragma unroll 1
    for (size_t axis = 0; axis < 3; ++axis) {
        if (voxel_vertex_indices(axis) == VERTEX_TO_ALLOCATE) {
            axis_offset(axis) = 1;
            Vector3i Xlocal_axis = Xlocal + axis_offset;
            Vector3i dXsv_axis = NeighborOffsetOfBoundaryVoxel(Xlocal_axis);
            int neighbor_idx = LinearizeNeighborOffset(dXsv_axis);

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
            assert(cached_subvolumes[neighbor_idx] != nullptr);
#endif

            float tsdf_axis = cached_subvolumes[neighbor_idx]->tsdf(
                Xlocal_axis - float(N) * dXsv_axis);

            float mu = (0 - tsdf_0) / (tsdf_axis - tsdf_0);

            Vector3f Xlocal_interp_axis = tsdf_volume.voxelf_local_to_global(
                Vector3f(Xlocal(0) + mu * axis_offset(0),
                         Xlocal(1) + mu * axis_offset(1),
                         Xlocal(2) + mu * axis_offset(2)),
                Xsv);

            voxel_vertex_indices(axis) = mesh_.vertices().push_back(
                tsdf_volume.voxelf_to_world(Xlocal_interp_axis));

            if (type & VertexWithNormal) {
                mesh_.vertex_normals()[voxel_vertex_indices(axis)] =
                    tsdf_volume.transform_volume_to_world_.Rotate(
                        (1 - mu) * gradient_0
                            + mu * tsdf_volume.gradient(
                                Xlocal_axis, cached_subvolumes));
            }

            axis_offset(axis) = 0;
        }
    }
}

template<VertexType type, size_t N>
__device__
void ScalableMeshVolumeCudaServer<type, N>::ExtractTriangle(
    const Vector3i &Xlocal, int subvolume_idx) {

    const uchar table_index = table_indices(Xlocal, subvolume_idx);
    if (table_index == 0 || table_index == 255) return;

    for (size_t tri = 0; tri < 16; tri += 3) {
        if (tri_table[table_index][tri] == -1) return;

        /** Edge index -> neighbor cube index ([0, 1])^3 x vertex index (3) **/
        Vector3i tri_vertex_indices;
#pragma unroll 1
        for (size_t vertex = 0; vertex < 3; ++vertex) {
            /** Edge holding the vetex **/
            int edge = tri_table[table_index][tri + vertex];
            /** Voxel holding the edge **/
            Vector3i Xlocal_edge_holder = Vector3i(
                Xlocal(0) + edge_shift[edge][0],
                Xlocal(1) + edge_shift[edge][1],
                Xlocal(2) + edge_shift[edge][2]);
            tri_vertex_indices(vertex) = vertex_indices(
                Xlocal_edge_holder, subvolume_idx)(edge_shift[edge][3]);
        }
        mesh_.triangles().push_back(tri_vertex_indices);
    }
}

template<VertexType type, size_t N>
__device__
void ScalableMeshVolumeCudaServer<type, N>::ExtractTriangleOnBoundary(
    const Vector3i &Xlocal, int subvolume_idx,
    int *cached_subvolume_indices) {

    const uchar table_index = table_indices(Xlocal, subvolume_idx);
    if (table_index == 0 || table_index == 255) return;

    for (size_t tri = 0; tri < 16; tri += 3) {
        if (tri_table[table_index][tri] == -1) return;

        /** Edge index -> neighbor cube index ([0, 1])^3 x vertex index (3) **/
        Vector3i tri_vertex_indices;
#pragma unroll 1
        for (size_t vertex = 0; vertex < 3; ++vertex) {
            /** Edge holding the vertex **/
            int edge = tri_table[table_index][tri + vertex];

            /** Voxel hoding the edge **/
            Vector3i Xlocal_edge_holder =
                Vector3i(Xlocal(0) + edge_shift[edge][0],
                         Xlocal(1) + edge_shift[edge][1],
                         Xlocal(2) + edge_shift[edge][2]);
            Vector3i dXsv_edge_holder =
                NeighborOffsetOfBoundaryVoxel(Xlocal_edge_holder);
            int neighbor_idx = LinearizeNeighborOffset(dXsv_edge_holder);
            int neighbor_subvolume_idx = cached_subvolume_indices[neighbor_idx];

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
            assert(neighbor_subvolume_idx != NULLPTR_CUDA);
#endif
            Vector3i Xlocal_edge_holder_in_neighbor =
                BoundaryVoxelInNeighbor(Xlocal_edge_holder, dXsv_edge_holder);
            tri_vertex_indices(vertex) = vertex_indices(
                Xlocal_edge_holder_in_neighbor, neighbor_subvolume_idx)(
                edge_shift[edge][3]);
        }
        mesh_.triangles().push_back(tri_vertex_indices);
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