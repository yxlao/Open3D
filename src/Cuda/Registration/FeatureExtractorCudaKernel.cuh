//
// Created by wei on 1/23/19.
//

#pragma once

#include "FeatureExtractorCuda.h"
#include "FeatureExtractorCudaDevice.cuh"

namespace open3d {
namespace cuda {
__global__
void ComputeSPFHFeatureKernel(FeatureCudaDevice server) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= server.neighbors_.indices_.size()) return;

    int i = server.neighbors_.indices_[idx];
    int max_nn = server.neighbors_.nn_count_[idx];
    server.ComputeSPFHFeature(i, max_nn);
}

void FeatureCudaKernelCaller::ComputeSPFHFeature(FeatureExtractorCuda &feature) {
    const dim3 blocks(DIV_CEILING(feature.neighbors_.indices_.size(),
        THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    ComputeSPFHFeatureKernel<<<blocks, threads>>>(*feature.server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void ComputeFPFHFeatureKernel(FeatureCudaDevice server) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= server.neighbors_.indices_.size()) return;

    int i = server.neighbors_.indices_[idx];
    int max_nn = server.neighbors_.nn_count_[idx];
    server.ComputeFPFHFeature(i, max_nn);
}

void FeatureCudaKernelCaller::ComputeFPFHFeature(FeatureExtractorCuda &feature) {
    const dim3 blocks(DIV_CEILING(feature.neighbors_.indices_.size(),
        THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    ComputeFPFHFeatureKernel<<<blocks, threads>>>(*feature.server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d