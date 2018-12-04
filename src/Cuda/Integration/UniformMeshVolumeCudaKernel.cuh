//
// Created by wei on 10/20/18.
//

#pragma once
#include "UniformMeshVolumeCudaDevice.cuh"

namespace open3d {
namespace cuda {
template<size_t N>
__global__
void MarchingCubesVertexAllocationKernel(
    UniformMeshVolumeCudaServer<N> server,
    UniformTSDFVolumeCudaServer<N> tsdf_volume) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= N - 1 || y >= N - 1 || z >= N - 1) return;

    Vector3i X = Vector3i(x, y, z);
    server.AllocateVertex(X, tsdf_volume);
}

template<size_t N>
__host__
void UniformMeshVolumeCudaKernelCaller<N>::
MarchingCubesVertexAllocationKernelCaller(
    UniformMeshVolumeCudaServer<N> &server,
    UniformTSDFVolumeCudaServer<N> &tsdf_volume) {

    const int num_blocks = DIV_CEILING(N, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    MarchingCubesVertexAllocationKernel << < blocks, threads >> > (
        server, tsdf_volume);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<size_t N>
__global__
void MarchingCubesVertexExtractionKernel(
    UniformMeshVolumeCudaServer<N> server,
    UniformTSDFVolumeCudaServer<N> tsdf_volume) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    Vector3i X = Vector3i(x, y, z);
    server.ExtractVertex(X, tsdf_volume);
}

template<size_t N>
__host__
void UniformMeshVolumeCudaKernelCaller<N>::
MarchingCubesVertexExtractionKernelCaller(
    UniformMeshVolumeCudaServer<N> &server,
    UniformTSDFVolumeCudaServer<N> &tsdf_volume) {

    const int num_blocks = DIV_CEILING(N, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    MarchingCubesVertexExtractionKernel << < blocks, threads >> > (
        server, tsdf_volume);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<size_t N>
__global__
void MarchingCubesTriangleExtractionKernel(
    UniformMeshVolumeCudaServer<N> server) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= N - 1 || y >= N - 1 || z >= N - 1) return;

    Vector3i X = Vector3i(x, y, z);
    server.ExtractTriangle(X);
}

template<size_t N>
__host__
void UniformMeshVolumeCudaKernelCaller<N>::
MarchingCubesTriangleExtractionKernelCaller(
    UniformMeshVolumeCudaServer<N> &server) {

    const int num_blocks = DIV_CEILING(N, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    MarchingCubesTriangleExtractionKernel << < blocks, threads >> > (server);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d