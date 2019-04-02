//
// Created by wei on 10/20/18.
//

#pragma once
#include "UniformMeshVolumeCudaDevice.cuh"

namespace open3d {
namespace cuda {

__global__
void VertexAllocationKernel(
    UniformMeshVolumeCudaDevice server,
    UniformTSDFVolumeCudaDevice tsdf_volume) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= server.N_ - 1 || y >= server.N_ - 1 || z >= server.N_ - 1)
        return;

    Vector3i X = Vector3i(x, y, z);
    server.AllocateVertex(X, tsdf_volume);
}


__host__
void UniformMeshVolumeCudaKernelCaller::VertexAllocation(
    UniformMeshVolumeCuda &mesher,
    UniformTSDFVolumeCuda &tsdf_volume) {

    const int num_blocks = DIV_CEILING(mesher.N_, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    VertexAllocationKernel << < blocks, threads >> > (
        *mesher.device_, *tsdf_volume.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}


__global__
void VertexExtractionKernel(
    UniformMeshVolumeCudaDevice server,
    UniformTSDFVolumeCudaDevice tsdf_volume) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    Vector3i X = Vector3i(x, y, z);
    server.ExtractVertex(X, tsdf_volume);
}


__host__
void UniformMeshVolumeCudaKernelCaller::VertexExtraction(
    UniformMeshVolumeCuda &mesher,
    UniformTSDFVolumeCuda &tsdf_volume) {

    const int num_blocks = DIV_CEILING(mesher.N_, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    VertexExtractionKernel << < blocks, threads >> > (
        *mesher.device_, *tsdf_volume.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}


__global__
void TriangleExtractionKernel(
    UniformMeshVolumeCudaDevice server) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= server.N_ - 1 || y >= server.N_ - 1 || z >= server.N_ - 1) return;

    Vector3i X = Vector3i(x, y, z);
    server.ExtractTriangle(X);
}


__host__
void UniformMeshVolumeCudaKernelCaller::TriangleExtraction(
    UniformMeshVolumeCuda &mesher) {

    const int num_blocks = DIV_CEILING(mesher.N_, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    TriangleExtractionKernel << < blocks, threads >> > (*mesher.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d