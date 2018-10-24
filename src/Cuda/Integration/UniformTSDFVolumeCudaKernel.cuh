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
    Vector3f n = server.RayCasting(x, y, camera, transform_camera_to_world);
    image.get(x, y) = (n == Vector3f::Zeros()) ? n : n * 0.5f + Vector3f(0.5f);
}
}