//
// Created by wei on 10/10/18.
//

#pragma once

#include "MarchingCubesConstCuda.h"
#include "UniformTSDFVolumeCuda.cuh"
#include <Cuda/Container/ArrayCuda.cuh>
#include <Cuda/Geometry/ImageCuda.cuh>

namespace open3d {
template<size_t N>
__global__
void IntegrateKernel(UniformTSDFVolumeCudaServer<N> server,
                     ImageCudaServer<Vector1f> depth,
                     MonoPinholeCameraCuda camera,
                     TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= N || y >= N || z >= N) return;

    /** Projective data association **/
    Vector3f X_w = server.voxel_to_world(x, y, z);
    Vector3f X_c = transform_camera_to_world.Inverse() * X_w;
    Vector2f p = camera.Projection(X_c);

    /** TSDF **/
    if (!camera.IsValid(p)) return;
    float d = depth.get_interp(p(0), p(1))(0);

    float sdf = d - X_c(2);
    if (sdf <= -server.sdf_trunc_) return;
    sdf = fminf(sdf, server.sdf_trunc_);

    /** Weight average **/
    float &tsdf = server.tsdf(x, y, z);
    float &weight = server.weight(x, y, z);

    /** TODO: color **/
    tsdf = (tsdf * weight + sdf * 1.0f) / (weight + 1.0f);
    weight += 1.0f;
}

template<size_t N>
__global__
void RayCastingKernel(UniformTSDFVolumeCudaServer<N> server,
                      ImageCudaServer<Vector3f> image,
                      MonoPinholeCameraCuda camera,
                      TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= image.width_ || y >= image.height_) return;
    image.get(x, y) = Vector3f(0);

    Vector3f ray_c = camera.InverseProjection(x, y, 1.0f).normalized();

    /** TODO: throw it into parameters **/
    const float t_min = 0.2f / ray_c(2);
    const float t_max = 3.0f / ray_c(2);

    const Vector3f camera_origin_v = server.transform_world_to_volume_ *
        (transform_camera_to_world * Vector3f(0));
    const Vector3f ray_v = server.transform_world_to_volume_.Rotate(
        transform_camera_to_world.Rotate(ray_c));

    float t_prev = 0, tsdf_prev = 0;

    /** Do NOT use #pragma unroll: it will make it slow **/
    float t_curr = t_min;
    while (t_curr < t_max) {
        Vector3f X_v = camera_origin_v + t_curr * ray_v;
        Vector3f X_voxel = server.volume_to_voxel(X_v);

        if (!server.InVolumef(X_voxel)) return;

        float tsdf_curr = server.tsdf(X_voxel);

        float step_size = tsdf_curr == 0 ?
                          server.sdf_trunc_ : fmaxf(tsdf_curr,
                                                    server.voxel_length_);

        if (tsdf_prev > 0 && tsdf_curr < 0) { /** Zero crossing **/
            float t_intersect = (t_curr * tsdf_prev - t_prev * tsdf_curr)
                / (tsdf_prev - tsdf_curr);

            Vector3f X_surface_v = camera_origin_v + t_intersect * ray_v;
            Vector3f X_surface_voxel = server.volume_to_voxel(X_surface_v);
            Vector3f normal_v = server.GradientAt(X_surface_voxel).normalized();
            image.get(x, y) = transform_camera_to_world.Inverse().Rotate(
                server.transform_volume_to_world_.Rotate(normal_v));
            return;
        }

        tsdf_prev = tsdf_curr;
        t_prev = t_curr;
        t_curr += step_size;
    }
}

__device__
inline Vector3f InterpVertex(const Vector3i &X0, float tsdf0,
                             const Vector3i &X1, float tsdf1) {
    float mu = (0 - tsdf0) / (tsdf1 - tsdf0);
    return Vector3f((1 - mu) * X0(0) + mu * X1(0),
                    (1 - mu) * X0(1) + mu * X1(1),
                    (1 - mu) * X0(2) + mu * X1(2));
}

__device__
inline Vector3i Shift(const Vector3i &X, const int i) {
    return Vector3i(X(0) + shift[i][0],
                    X(1) + shift[i][1],
                    X(2) + shift[i][2]);
}

template<size_t N>
__global__
void MarchingCubesVertexExtractionKernel(
    UniformTSDFVolumeCudaServer<N> server) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= N - 1 || y >= N - 1 || z >= N - 1) return;

    const Vector3i X_voxel = Vector3i(x, y, z);

    float tsdf[8];
    int &table_index = server.table_index(x, y, z);
    table_index = 0;

    int tmp_table_index = 0;
#pragma unroll 1
    for (int i = 0; i < 8; ++i) {
        const Vector3i Xi_voxel = Shift(X_voxel, i);

        float weight = server.weight(Xi_voxel);
        if (weight == 0) return;

        tsdf[i] = server.tsdf(Xi_voxel);
        if (fabsf(tsdf[i]) > server.voxel_length_) return;
        tmp_table_index |= ((tsdf[i] < 0) ? (1 << i) : 0);
    }
    if (tmp_table_index == 0 || tmp_table_index == 255) return;
    table_index = tmp_table_index;

    /** Extract up to 3 edges in this voxel: 0 - 1; 0 - 3; 0 - 4 **/
    int edges = edge_table[table_index];
#pragma unroll 12
    for (int i = 0; i < 12; ++i) {
        if (edges & (1 << i)) {
            Vector3i Xi_voxel = Vector3i(x + edge_shift[i][0],
                                         y + edge_shift[i][1],
                                         z + edge_shift[i][2]);
            int idx = edge_shift[i][3];
            int lock = atomicExch(&(server.vertex_locks(Xi_voxel)(idx)), LOCKED);
            if (lock != LOCKED) {
                Vector3f Xi_voxel_interp = InterpVertex(
                    Shift(X_voxel, edge_to_vert[i][0]), tsdf[edge_to_vert[i][0]],
                    Shift(X_voxel, edge_to_vert[i][1]), tsdf[edge_to_vert[i][1]]);
                server.vertex_indices(Xi_voxel)(idx) = server.mesh().vertices().push_back(
                    server.voxel_to_world(Xi_voxel_interp));
            }
        }
    }
}

template<size_t N>
__global__
void MarchingCubesTriangleExtractionKernel(
    UniformTSDFVolumeCudaServer<N> server) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= N - 1 || y >= N - 1 || z >= N - 1) return;

    /** Get table index - it is always faster to recompute it rather than
     * save it **/
    const int table_index = server.table_index(x, y, z);
    if (table_index == 0 || table_index == 255) return;

#pragma unroll 1
    for (int i = 0; i < 16; i += 3) {
        if (tri_table[table_index][i] == -1) return;

        int edge0 = tri_table[table_index][i + 0];
        int edge1 = tri_table[table_index][i + 1];
        int edge2 = tri_table[table_index][i + 2];

        Vector3i vertex_indices;
        vertex_indices(0) = server.vertex_indices(x + edge_shift[edge0][0],
                                                  y + edge_shift[edge0][1],
                                                  z + edge_shift[edge0][2])(
            edge_shift[edge0][3]);

        vertex_indices(1) = server.vertex_indices(x + edge_shift[edge1][0],
                                                  y + edge_shift[edge1][1],
                                                  z + edge_shift[edge1][2])(
            edge_shift[edge1][3]);

        vertex_indices(2) = server.vertex_indices(x + edge_shift[edge2][0],
                                                  y + edge_shift[edge2][1],
                                                  z + edge_shift[edge2][2])(
            edge_shift[edge2][3]);
        server.mesh().triangles().push_back(vertex_indices);
    }
}
}