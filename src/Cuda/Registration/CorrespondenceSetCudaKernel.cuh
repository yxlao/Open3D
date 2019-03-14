//
// Created by wei on 1/14/19.
//

#pragma once

#include "CorrespondenceSetCuda.h"
#include <src/Cuda/Common/UtilsCuda.h>
#include <src/Cuda/Container/ArrayCudaDevice.cuh>
#include <src/Cuda/Container/Array2DCudaDevice.cuh>

namespace open3d {
namespace cuda {

__global__
void CompressCorrespondencesKernel(CorrespondenceSetCudaDevice corres) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= corres.matrix_.max_cols_) return;

    int nn = 0;
    for (int j = 0; j < corres.matrix_.max_rows_; ++j) {
        if (corres.matrix_(j, i) == -1) break;
        ++nn;
    }

    if (nn > 0) {
        int addr = corres.indices_.push_back(i);
        corres.nn_count_[addr] = nn; // Trick for asynchronization
    }
}

void CorrespondenceSetCudaKernelCaller::CompressCorrespondence(
    CorrespondenceSetCuda &corres) {

    corres.indices_.set_iterator(0);

    const dim3 blocks(DIV_CEILING(corres.matrix_.max_cols_, THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    CompressCorrespondencesKernel << < blocks, threads >> > (*corres.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    corres.nn_count_.set_iterator(corres.indices_.size());
}
}
}