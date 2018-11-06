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
                     RGBDImageCudaServer rgbd,
                     MonoPinholeCameraCuda camera,
                     TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= N || y >= N || z >= N) return;

    Vector3i X = Vector3i(x, y, z);
    server.Integrate(X, rgbd, camera, transform_camera_to_world);
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

    Vector2i p = Vector2i(x, y);
    Vector3f n = server.RayCasting(p, camera, transform_camera_to_world);

    image.get(x, y) = (n == Vector3f::Zeros()) ?
        n : Vector3f((n(0) + 1) * 0.5f, (n(1) + 1) * 0.5f, (n(2) + 1) * 0.5f);
}
}