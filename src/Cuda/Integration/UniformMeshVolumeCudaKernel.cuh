//
// Created by wei on 10/20/18.
//

#include "UniformMeshVolumeCuda.cuh"

namespace open3d {
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
}