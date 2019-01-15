//
// Created by wei on 1/14/19.
//

#pragma once

#include "CorrespondenceSetCuda.h"
#include <Cuda/Common/UtilsCuda.h>
#include <Cuda/Container/ArrayCudaDevice.cuh>
#include <Cuda/Container/MatrixCudaDevice.cuh>
#include <Core/Core.h>

namespace open3d {
namespace cuda {

__global__
void CompressCorrespondencesKernel(
    MatrixCudaDevice<int> corres_matrix,
    ArrayCudaDevice<int> corres_indices) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < corres_matrix.max_rows_ && corres_matrix(i, 0) != -1) {
        corres_indices.push_back(i);
    }
}

void CorrespondenceSetCudaKernelCaller::CompressCorrespondenceKernelCaller(
    MatrixCuda<int> &corres_matrix,
    ArrayCuda<int> &corres_indices) {

    corres_indices.set_iterator(0);

    const dim3 blocks(DIV_CEILING(corres_matrix.max_rows_, THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    CompressCorrespondencesKernel<<<blocks, threads>>>(
        *corres_matrix.server(), *corres_indices.server());
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
}
}