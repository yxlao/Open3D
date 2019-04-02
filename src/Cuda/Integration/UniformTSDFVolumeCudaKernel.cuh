//
// Created by wei on 10/10/18.
//

#pragma once

#include "UniformTSDFVolumeCudaDevice.cuh"

namespace open3d {
namespace cuda {

__global__
void IntegrateKernel(UniformTSDFVolumeCudaDevice server,
                     RGBDImageCudaDevice rgbd,
                     PinholeCameraIntrinsicCuda camera,
                     TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= server.N_ || y >= server.N_ || z >= server.N_) return;

    Vector3i X = Vector3i(x, y, z);
    server.Integrate(X, rgbd, camera, transform_camera_to_world);
}


__host__
void UniformTSDFVolumeCudaKernelCaller::Integrate(
    UniformTSDFVolumeCuda &volume,
    RGBDImageCuda &rgbd,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {

    const int num_blocks = DIV_CEILING(volume.N_, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    IntegrateKernel << < blocks, threads >> > (
        *volume.device_, *rgbd.device_, camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}


__global__
void RayCastingKernel(UniformTSDFVolumeCudaDevice server,
                      ImageCudaDevice<float, 3> image,
                      PinholeCameraIntrinsicCuda camera,
                      TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= image.width_ || y >= image.height_) return;

    Vector2i p = Vector2i(x, y);
    Vector3f n = server.RayCasting(p, camera, transform_camera_to_world);

    image.at(x, y) = (n == Vector3f::Zeros()) ?
                     n : Vector3f((n(0) + 1) * 0.5f,
                                  (n(1) + 1) * 0.5f,
                                  (n(2) + 1) * 0.5f);
}


void UniformTSDFVolumeCudaKernelCaller::RayCasting(
    UniformTSDFVolumeCuda &volume,
    ImageCuda<float, 3> &image,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {

    const dim3 blocks(DIV_CEILING(image.width_, THREAD_2D_UNIT),
                      DIV_CEILING(image.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    RayCastingKernel << < blocks, threads >> > (
        *volume.device_, *image.device_, camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d