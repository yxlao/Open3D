//
// Created by wei on 11/2/18.
//

#pragma once

#include "TriangleMeshCuda.h"
#include <src/Cuda/Container/ArrayCudaDevice.cuh>

namespace open3d {
namespace cuda {
__global__
void GetMinBoundKernel(TriangleMeshCudaDevice mesh,
                       ArrayCudaDevice<Vector3f> min_bound) {
    __shared__ float local_min_x[THREAD_1D_UNIT];
    __shared__ float local_min_y[THREAD_1D_UNIT];
    __shared__ float local_min_z[THREAD_1D_UNIT];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    Vector3f vertex = idx < mesh.vertices_.size() ?
                      mesh.vertices_[idx] : Vector3f(1e10f);

    local_min_x[tid] = vertex(0);
    local_min_y[tid] = vertex(1);
    local_min_z[tid] = vertex(2);
    __syncthreads();

    TripleBlockReduceMin<float>(local_min_x, local_min_y, local_min_z, tid);

    if (tid == 0) {
        atomicMinf(&(min_bound[0](0)), local_min_x[0]);
        atomicMinf(&(min_bound[0](1)), local_min_y[0]);
        atomicMinf(&(min_bound[0](2)), local_min_z[0]);
    }
}

__host__
void TriangleMeshCudaKernelCaller::GetMinBound(
    const TriangleMeshCuda &mesh, ArrayCuda<Vector3f> &min_bound) {
    const dim3 blocks(DIV_CEILING(mesh.vertices_.size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    GetMinBoundKernel << < blocks, threads >> > (
        *mesh.device_, *min_bound.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void GetMaxBoundKernel(TriangleMeshCudaDevice mesh,
                       ArrayCudaDevice<Vector3f> max_bound) {
    __shared__ float local_max_x[THREAD_1D_UNIT];
    __shared__ float local_max_y[THREAD_1D_UNIT];
    __shared__ float local_max_z[THREAD_1D_UNIT];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    Vector3f vertex = idx < mesh.vertices_.size() ?
                      mesh.vertices_[idx] : Vector3f(-1e10f);

    local_max_x[tid] = vertex(0);
    local_max_y[tid] = vertex(1);
    local_max_z[tid] = vertex(2);
    __syncthreads();

    TripleBlockReduceMax<float>(local_max_x, local_max_y, local_max_z, tid);

    if (tid == 0) {
        atomicMaxf(&(max_bound[0](0)), local_max_x[0]);
        atomicMaxf(&(max_bound[0](1)), local_max_y[0]);
        atomicMaxf(&(max_bound[0](2)), local_max_z[0]);
    }
}

__host__
void TriangleMeshCudaKernelCaller::GetMaxBound(
    const TriangleMeshCuda &mesh, ArrayCuda<Vector3f> &max_bound) {

    const dim3 blocks(DIV_CEILING(mesh.vertices_.size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    GetMaxBoundKernel << < blocks, threads >> > (
        *mesh.device_, *max_bound.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void TransformKernel(TriangleMeshCudaDevice mesh, TransformCuda transform) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= mesh.vertices_.size()) return;

    Vector3f &vertex_position = mesh.vertices_[idx];
    vertex_position = transform * vertex_position;

    if (mesh.type_ & VertexWithNormal) {
        Vector3f &vertex_normal = mesh.vertex_normals_[idx];
        vertex_normal = transform.Rotate(vertex_normal);
    }
}

__host__
void TriangleMeshCudaKernelCaller::Transform(
    TriangleMeshCuda &mesh, TransformCuda &transform) {

    const dim3 blocks(DIV_CEILING(mesh.vertices_.size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    TransformKernel << < blocks, threads >> > (*mesh.device_, transform);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d