//
// Created by wei on 2/12/19.
//

#pragma once

#include "RGBDImageCuda.h"
#include "ImageCudaDevice.cuh"

namespace open3d {

namespace cuda {
__global__
void ConvertDepthToFloatKernel(RGBDImageCudaDevice device,
                               float factor, float depth_trunc) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= device.width_ || v >= device.height_) return;

    float d = device.depth_raw_.at(u, v)(0) / factor;
    device.depth_.at(u, v)(0) = d >= depth_trunc ? 0.0f : d;
}

void RGBDImageCudaKernelCaller::ConvertDepthToFloat(RGBDImageCuda &rgbd) {
    const dim3 blocks(DIV_CEILING(rgbd.width_, THREAD_2D_UNIT),
                      DIV_CEILING(rgbd.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ConvertDepthToFloatKernel << < blocks, threads >> > (
        *rgbd.device_, rgbd.depth_factor_, rgbd.depth_trunc_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // namespace