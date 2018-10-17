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
    server.Integrate(x, y, z, depth, camera, transform_camera_to_world);
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
    image.get(x, y) = server.RayCasting(
        x, y, camera, transform_camera_to_world);
}

template<size_t N>
__global__
void MarchingCubesVertexAllocationKernel(
    UniformTSDFVolumeCudaServer<N> server) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= N - 1 || y >= N - 1 || z >= N - 1) return;

    const Vector3i X_voxel = Vector3i(x, y, z);

    uchar &table_index = server.table_index(x, y, z);
    table_index = 0;

    int tmp_table_index = 0;

    /** There are early returns. #pragma unroll SLOWS it down **/
    for (int i = 0; i < 8; ++i) {
        const int xi = x + shift[i][0] ;
        const int yi = y + shift[i][1];
        const int zi = z + shift[i][2];

        uchar weight = server.weight(xi, yi, zi);
        if (weight == 0) return;

        float tsdf = server.tsdf(xi, yi, zi);
        if (fabsf(tsdf) > 2 * server.voxel_length_) return;

        tmp_table_index |= ((tsdf < 0) ? (1 << i) : 0);
    }
    if (tmp_table_index == 0 || tmp_table_index == 255) return;
    table_index = (uchar)tmp_table_index;

    /** Extract up to 3 edges in this voxel: 0 - 1; 0 - 3; 0 - 4 **/
    int edges = edge_table[table_index];
#pragma unroll 12
    for (int i = 0; i < 12; ++i) {
        if (edges & (1 << i)) {
            server.vertex_indices(x + edge_shift[i][0],
                                  y + edge_shift[i][1],
                                  z + edge_shift[i][2])
                (edge_shift[i][3]) = 0;
        }
    }
}

template<size_t N>
__global__
void MarchingCubesVertexExtractionKernel(
    UniformTSDFVolumeCudaServer<N> server) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= N || y >= N || z >= N) return;

    Vector3i &vertex_index = server.vertex_indices(x, y, z);
    bool f1 = vertex_index(0) == 0;
    bool f2 = vertex_index(1) == 0;
    bool f3 = vertex_index(2) == 0;
    if (!f1 && !f2 && !f3) return;

    const Vector3i X_voxel = Vector3i(x, y, z);
    float tsdf0 = server.tsdf(x, y, z);

    if (f1) {
        float tsdf1 = server.tsdf(x + 1, y, z);
        float mu = (0 - tsdf0) / (tsdf1 - tsdf0);
        vertex_index(0) = server.mesh().vertices().push_back(
            server.voxel_to_world(x + mu, y, z));
        server.transform_volume_to_world_.Rotate(
            (1 - mu) * server.gradient(x, y, z) + mu * server.gradient(x + 1, y,
                                                                       z));
    }

    if (f2) {
        float tsdf3 = server.tsdf(x, y + 1, z);
        float mu = (0 - tsdf0) / (tsdf3 - tsdf0);
        vertex_index(1) = server.mesh().vertices().push_back(
            server.voxel_to_world(x, y + mu, z));
        server.transform_volume_to_world_.Rotate(
            (1 - mu) * server.gradient(x, y, z) + mu * server.gradient(x, y + 1,
                                                                       z));
    }
    if (f3) {
        float tsdf4 = server.tsdf(x, y, z + 1);
        float mu = (0 - tsdf0) / (tsdf4 - tsdf0);
        vertex_index(2) = server.mesh().vertices().push_back(
            server.voxel_to_world(x, y, z + mu));
        server.transform_volume_to_world_.Rotate(
            (1 - mu) * server.gradient(x, y, z) + mu * server.gradient(x, y, z +
                1));
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
    const uchar table_index = server.table_index(x, y, z);
    if (table_index == 0 || table_index == 255) return;

#pragma unroll 1
    for (int i = 0; i < 16; i += 3) {
        if (tri_table[table_index][i] == -1) return;

        int edge0 = tri_table[table_index][i + 0];
        int edge1 = tri_table[table_index][i + 1];
        int edge2 = tri_table[table_index][i + 2];

        Vector3i vertex_indices;

        vertex_indices(0) = server.vertex_indices(
            x + edge_shift[edge0][0],
            y + edge_shift[edge0][1],
            z + edge_shift[edge0][2])(edge_shift[edge0][3]);

        vertex_indices(1) = server.vertex_indices(
            x + edge_shift[edge1][0],
            y + edge_shift[edge1][1],
            z + edge_shift[edge1][2])(edge_shift[edge1][3]);

        vertex_indices(2) = server.vertex_indices(
            x + edge_shift[edge2][0],
            y + edge_shift[edge2][1],
            z + edge_shift[edge2][2])(edge_shift[edge2][3]);

        server.mesh().triangles().push_back(vertex_indices);
    }
}
}