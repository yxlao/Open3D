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

            /**
             * Image coordinate system is different from OpenGL,
             * so clockwise becomes counter-clockwise.
             * Native Open3d use this: 0 1 2 -> 0 2 1
             * in ScalableTSDFVolume @ExtractTriangleMesh
             * Similarly, we use this: 0 1 2 -> 2 1 0.
             */
            tri_vertex_indices(2 - vertex) = vertex_indices(
                X_edge_holder)(edge_shift[edge][3]);
        }
        mesh_.triangles().push_back(tri_vertex_indices);
    }
}
}