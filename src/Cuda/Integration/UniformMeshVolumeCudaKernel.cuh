//
// Created by wei on 10/20/18.
//

#include "UniformMeshVolumeCuda.cuh"

namespace open3d {
template<VertexType type, size_t N>
__global__
void MarchingCubesVertexAllocationKernel(
    UniformMeshVolumeCudaServer<type, N> server,
    UniformTSDFVolumeCudaServer<N> tsdf_volume) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= N - 1 || y >= N - 1 || z >= N - 1) return;

    server.AllocateVertex(x, y, z, tsdf_volume);
}

template<VertexType type, size_t N>
__global__
void MarchingCubesVertexExtractionKernel(
    UniformMeshVolumeCudaServer<type, N> server,
    UniformTSDFVolumeCudaServer<N> tsdf_volume) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= N || y >= N || z >= N) return;

    server.ExtractVertex(x, y, z, tsdf_volume);
}

template<VertexType type, size_t N>
__global__
void MarchingCubesTriangleExtractionKernel(
    UniformMeshVolumeCudaServer<type, N> server) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= N - 1 || y >= N - 1 || z >= N - 1) return;

    server.ExtractTriangle(x, y, z);
}
}