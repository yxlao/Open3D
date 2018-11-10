//
// Created by wei on 11/2/18.
//

#pragma once

#include "TriangleMeshCuda.h"
#include <Cuda/Container/ArrayCudaDevice.cuh>

namespace open3d {
__global__
void GetMinBoundKernel(TriangleMeshCudaServer server,
                       ArrayCudaServer<Vector3f> min_bound) {
    __shared__ float local_min_x[THREAD_1D_UNIT];
    __shared__ float local_min_y[THREAD_1D_UNIT];
    __shared__ float local_min_z[THREAD_1D_UNIT];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    Vector3f vertex = idx < server.vertices().size() ?
        server.vertices()[idx] : Vector3f(1e10f);

    local_min_x[tid] = vertex(0);
    local_min_y[tid] = vertex(1);
    local_min_z[tid] = vertex(2);
    __syncthreads();

    if (tid < 32) {
        WarpReduceMin(local_min_x, tid);
        WarpReduceMin(local_min_y, tid);
        WarpReduceMin(local_min_z, tid);
    }
    __syncthreads();

    if (tid == 0) {
        atomicMinf(&(min_bound[0](0)), local_min_x[0]);
        atomicMinf(&(min_bound[0](1)), local_min_y[0]);
        atomicMinf(&(min_bound[0](2)), local_min_z[0]);
    }
}

__host__
void TriangleMeshCudaKernelCaller::GetMinBoundKernelCaller(
    TriangleMeshCudaServer &server,
    ArrayCudaServer<Vector3f> &min_bound,
    int num_vertices) {
    const dim3 blocks(DIV_CEILING(num_vertices, THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    GetMinBoundKernel << < blocks, threads >> > (server, min_bound);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}


__global__
void GetMaxBoundKernel(TriangleMeshCudaServer server,
                       ArrayCudaServer<Vector3f> max_bound) {
    __shared__ float local_max_x[THREAD_1D_UNIT];
    __shared__ float local_max_y[THREAD_1D_UNIT];
    __shared__ float local_max_z[THREAD_1D_UNIT];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    Vector3f vertex = idx < server.vertices().size() ?
                      server.vertices()[idx] : Vector3f(-1e10f);

    local_max_x[tid] = vertex(0);
    local_max_y[tid] = vertex(1);
    local_max_z[tid] = vertex(2);
    __syncthreads();

    if (tid < 32) {
        WarpReduceMax(local_max_x, tid);
        WarpReduceMax(local_max_y, tid);
        WarpReduceMax(local_max_z, tid);
    }
    __syncthreads();

    if (tid == 0) {
        atomicMaxf(&(max_bound[0](0)), local_max_x[0]);
        atomicMaxf(&(max_bound[0](1)), local_max_y[0]);
        atomicMaxf(&(max_bound[0](2)), local_max_z[0]);
    }
}

__host__
void TriangleMeshCudaKernelCaller::GetMaxBoundKernelCaller(
    TriangleMeshCudaServer &server,
    ArrayCudaServer<Vector3f> &max_bound,
    int num_vertices) {

    const dim3 blocks(DIV_CEILING(num_vertices, THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    GetMaxBoundKernel << < blocks, threads >> > (server, max_bound);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void TransformKernel(TriangleMeshCudaServer server, TransformCuda transform) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= server.vertices().size()) return;

    Vector3f &vertex_position = server.vertices()[idx];
    vertex_position = transform * vertex_position;

    if (server.type_ & VertexWithNormal) {
        Vector3f &vertex_normal = server.vertex_normals()[idx];
        vertex_normal = transform.Rotate(vertex_normal);
    }
}

__host__
void TriangleMeshCudaKernelCaller::TransformKernelCaller(
    TriangleMeshCudaServer& server,
    TransformCuda &transform,
    int num_vertices) {

    const dim3 blocks(DIV_CEILING(num_vertices, THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    TransformKernel << < blocks, threads >> > (server, transform);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
}