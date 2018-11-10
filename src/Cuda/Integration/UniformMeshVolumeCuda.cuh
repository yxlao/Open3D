//
// Created by wei on 10/16/18.
//

#pragma once

#include "UniformMeshVolumeCuda.h"
#include "MarchingCubesConstCuda.h"

#include "UniformTSDFVolumeCudaDevice.cuh"

#include <Core/Core.h>

namespace open3d {
/**
 * Server end
 */
template<size_t N>
__device__
void UniformMeshVolumeCudaServer<N>::AllocateVertex(
    const Vector3i &X,
    UniformTSDFVolumeCudaServer<N> &tsdf_volume) {

    uchar &table_index = table_indices(X);
    table_index = 0;

    int tmp_table_index = 0;

    /** There are early returns. #pragma unroll SLOWS it down **/
    for (size_t corner = 0; corner < 8; ++corner) {
        Vector3i X_corner = Vector3i(X(0) + shift[corner][0],
                                     X(1) + shift[corner][1],
                                     X(2) + shift[corner][2]);

        uchar weight = tsdf_volume.weight(X_corner);
        if (weight == 0) return;

        float tsdf = tsdf_volume.tsdf(X_corner);
        if (fabsf(tsdf) > tsdf_volume.sdf_trunc_) return;

        tmp_table_index |= ((tsdf < 0) ? (1 << corner) : 0);
    }
    if (tmp_table_index == 0 || tmp_table_index == 255) return;
    table_index = (uchar) tmp_table_index;

    /** Tell them they will be extracted. Conflict can be ignored **/
    int edges = edge_table[table_index];
#pragma unroll 12
    for (size_t edge = 0; edge < 12; ++edge) {
        if (edges & (1 << edge)) {
            Vector3i X_edge_holder = Vector3i(X(0) + edge_shift[edge][0],
                                              X(1) + edge_shift[edge][1],
                                              X(2) + edge_shift[edge][2]);
            vertex_indices(X_edge_holder)(edge_shift[edge][3]) =
                VERTEX_TO_ALLOCATE;
        }
    }
}

template<size_t N>
__device__
void UniformMeshVolumeCudaServer<N>::ExtractVertex(
    const Vector3i &X,
    UniformTSDFVolumeCudaServer<N> &tsdf_volume) {

    Vector3i &voxel_vertex_indices = vertex_indices(X);
    if (voxel_vertex_indices(0) != VERTEX_TO_ALLOCATE
        && voxel_vertex_indices(1) != VERTEX_TO_ALLOCATE
        && voxel_vertex_indices(2) != VERTEX_TO_ALLOCATE)
        return;

    Vector3i axis_offset = Vector3i::Zeros();

    float tsdf_0 = tsdf_volume.tsdf(X);

    Vector3f gradient_0 = (mesh_.type_ & VertexWithNormal) ?
        tsdf_volume.gradient(X) : Vector3f::Zeros();

    Vector3b color_0 = (mesh_.type_ & VertexWithColor) ?
        tsdf_volume.color(X) : Vector3b::Zeros();

#pragma unroll 1
    for (size_t axis = 0; axis < 3; ++axis) {
        if (voxel_vertex_indices(axis) == VERTEX_TO_ALLOCATE) {
            axis_offset(axis) = 1;
            Vector3i X_axis = X + axis_offset;

            float tsdf_axis = tsdf_volume.tsdf(X_axis);
            float mu = (0 - tsdf_0) / (tsdf_axis - tsdf_0);

            voxel_vertex_indices(axis) = mesh_.vertices().push_back(
                tsdf_volume.voxelf_to_world(
                    Vector3f(X(0) + mu * axis_offset(0),
                             X(1) + mu * axis_offset(1),
                             X(2) + mu * axis_offset(2))));

            /** Note we share the vertex indices **/
            if (mesh_.type_ & VertexWithNormal) {
                mesh_.vertex_normals()[voxel_vertex_indices(axis)] =
                    tsdf_volume.transform_volume_to_world_.Rotate(
                        (1 - mu) * gradient_0
                            + mu * tsdf_volume.gradient(X_axis)).normalized();
            }

            if (mesh_.type_ & VertexWithColor) {
                assert(mu >= 0 && mu <= 1);
                Vector3b &color_axis = tsdf_volume.color(X_axis);
                mesh_.vertex_colors()[voxel_vertex_indices(axis)] = Vector3f(
                    ((1 - mu) * color_0(0) + mu * color_axis(0)) / 255.0f,
                    ((1 - mu) * color_0(1) + mu * color_axis(1)) / 255.0f,
                    ((1 - mu) * color_0(2) + mu * color_axis(2)) / 255.0f);
            }

            axis_offset(axis) = 0;
        }
    }
}

template<size_t N>
__device__
inline void UniformMeshVolumeCudaServer<N>::ExtractTriangle(
    const Vector3i &X) {

    const uchar table_index = table_indices(X);
    if (table_index == 0 || table_index == 255) return;

    for (size_t tri = 0; tri < 16; tri += 3) {
        if (tri_table[table_index][tri] == -1) return;

        /** Edge index -> neighbor cube index ([0, 1])^3 x vertex index (3) **/
        Vector3i tri_vertex_indices;
#pragma unroll 1
        for (int vertex = 0; vertex < 3; ++vertex) {
            /** Edge holding the vertex **/
            int edge = tri_table[table_index][tri + vertex];
            /** Voxel hoding the edge **/
            Vector3i X_edge_holder = Vector3i(X(0) + edge_shift[edge][0],
                                              X(1) + edge_shift[edge][1],
                                              X(2) + edge_shift[edge][2]);
            tri_vertex_indices(vertex) = vertex_indices(
                X_edge_holder)(edge_shift[edge][3]);
        }
        mesh_.triangles().push_back(tri_vertex_indices);
    }
}

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

    server_ = std::make_shared<UniformMeshVolumeCudaServer<N>>();

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

template<size_t N>
void UniformMeshVolumeCuda<N>::VertexExtraction(
    UniformTSDFVolumeCuda<N> &tsdf_volume) {
    assert(server_ != nullptr);

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

template<size_t N>
void UniformMeshVolumeCuda<N>::TriangleExtraction() {
    assert(server_ != nullptr);

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