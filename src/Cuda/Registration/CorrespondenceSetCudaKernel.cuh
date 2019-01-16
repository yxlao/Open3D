//
// Created by wei on 1/14/19.
//

#pragma once

#include "CorrespondenceSetCuda.h"
#include <Cuda/Common/UtilsCuda.h>
#include <Cuda/Container/ArrayCudaDevice.cuh>
#include <Cuda/Container/Array2DCudaDevice.cuh>
#include <Core/Core.h>

namespace open3d {
namespace cuda {

__global__
void CompressCorrespondencesKernel(CorrespondenceSetCudaDevice corres) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < corres.matrix_.max_rows_ && corres.matrix_(i, 0) != -1) {
        corres.indices_.push_back(i);
    }
}

void CorrespondenceSetCudaKernelCaller::CompressCorrespondenceKernelCaller(
    CorrespondenceSetCuda &corres) {

    corres.indices_.set_iterator(0);

    const dim3 blocks(DIV_CEILING(
                          corres.matrix_.max_rows_, THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    CompressCorrespondencesKernel << < blocks, threads >> > (*corres.server());
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
}
}