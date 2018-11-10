//
// Created by wei on 10/10/18.
//

#pragma once

#include "UniformTSDFVolumeCudaDevice.cuh"

namespace open3d {
template<size_t N>
__global__
void IntegrateKernel(UniformTSDFVolumeCudaServer<N> server,
                     RGBDImageCudaServer rgbd,
                     PinholeCameraIntrinsicCuda camera,
                     TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= N || y >= N || z >= N) return;

    Vector3i X = Vector3i(x, y, z);
    server.Integrate(X, rgbd, camera, transform_camera_to_world);
}

template<size_t N>
__host__
void UniformTSDFVolumeCudaKernelCaller<N>::IntegrateKernelCaller(
    UniformTSDFVolumeCudaServer<N> &server,
    RGBDImageCudaServer &rgbd,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {

    const int num_blocks = DIV_CEILING(N, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    IntegrateKernel << < blocks, threads >> > (
        server, rgbd, camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}


template<size_t N>
__global__
void RayCastingKernel(UniformTSDFVolumeCudaServer<N> server,
                      ImageCudaServer<Vector3f> image,
                      PinholeCameraIntrinsicCuda camera,
                      TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= image.width_ || y >= image.height_) return;

    Vector2i p = Vector2i(x, y);
    Vector3f n = server.RayCasting(p, camera, transform_camera_to_world);

    image.at(x, y) = (n == Vector3f::Zeros()) ?
        n : Vector3f((n(0) + 1) * 0.5f, (n(1) + 1) * 0.5f, (n(2) + 1) * 0.5f);
}

template<size_t N>
void UniformTSDFVolumeCudaKernelCaller<N>::RayCastingKernelCaller(
    UniformTSDFVolumeCudaServer<N> &server,
    ImageCudaServer<Vector3f> &image,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {

    const dim3 blocks(DIV_CEILING(image.width_, THREAD_2D_UNIT),
                      DIV_CEILING(image.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    RayCastingKernel << < blocks, threads >> > (
        server, image, camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
}