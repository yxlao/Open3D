//
// Created by wei on 4/4/19.
//

#pragma once

#include "ScalableVolumeRegistrationCuda.h"
#include "ScalableVolumeRegistrationCudaDevice.cuh"
#include <Cuda/Common/JacobianCuda.h>
#include <Cuda/Common/ReductionCuda.h>

namespace open3d {
namespace cuda {

__global__
void BuildLinearSystemKernel(
    ScalableVolumeRegistrationCudaDevice registration) {
    __shared__ float local_sum0[512];
    __shared__ float local_sum1[512];
    __shared__ float local_sum2[512];

    __shared__ UniformTSDFVolumeCudaDevice *subvolume;
    const int subvolume_idx = blockIdx.x;

    const HashEntry<Vector3i> &subvolume_entry =
        registration.source_.active_subvolume_entry_array_[subvolume_idx];

    const Vector3i Xsv = subvolume_entry.key;
    const Vector3i Xlocal = Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);
    const int tid = threadIdx.x + threadIdx.y * blockDim.x
                  + threadIdx.z * (blockDim.x * blockDim.y);
    if (tid == 0) {
        subvolume = registration.source_.QuerySubvolume(Xsv);
    }
    __syncthreads();

    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    HessianCuda<6> JtJ;
    Vector6f jacobian, Jtr;
    float residual;
    bool mask = registration.ComputeVoxelwiseJacobianAndResidual(
        Xlocal, Xsv, subvolume, subvolume_idx, jacobian, residual);
    if (mask) {
        ComputeJtJAndJtr(jacobian, residual, JtJ, Jtr);
    }

    for (size_t i = 0; i < 21; i += 3) {
        local_sum0[tid] = mask ? JtJ(i + 0) : 0;
        local_sum1[tid] = mask ? JtJ(i + 1) : 0;
        local_sum2[tid] = mask ? JtJ(i + 2) : 0;
        __syncthreads();

        BlockReduceSum<float, 512>(tid, local_sum0, local_sum1, local_sum2);

        if (tid == 0) {
            atomicAdd(&registration.results_.at(i + 0), local_sum0[0]);
            atomicAdd(&registration.results_.at(i + 1), local_sum1[0]);
            atomicAdd(&registration.results_.at(i + 2), local_sum2[0]);
        }
        __syncthreads();
    }

    /** Reduce Sum Jtr **/
    const int OFFSET1 = 21;
    for (size_t i = 0; i < 6; i += 3) {
        local_sum0[tid] = mask ? Jtr(i + 0) : 0;
        local_sum1[tid] = mask ? Jtr(i + 1) : 0;
        local_sum2[tid] = mask ? Jtr(i + 2) : 0;
        __syncthreads();

        BlockReduceSum<float, 512>(tid, local_sum0, local_sum1, local_sum2);

        if (tid == 0) {
            atomicAdd(&registration.results_.at(i + 0 + OFFSET1), local_sum0[0]);
            atomicAdd(&registration.results_.at(i + 1 + OFFSET1), local_sum1[0]);
            atomicAdd(&registration.results_.at(i + 2 + OFFSET1), local_sum2[0]);
        }
        __syncthreads();
    }

    /** Reduce Sum loss and inlier **/
    const int OFFSET2 = 27;
    {
        local_sum0[tid] = mask ? residual * residual : 0;
        local_sum1[tid] = mask ? 1 : 0;
        __syncthreads();

        BlockReduceSum<float, 512>(tid, local_sum0, local_sum1);

        if (tid == 0) {
            atomicAdd(&registration.results_.at(0 + OFFSET2), local_sum0[0]);
            atomicAdd(&registration.results_.at(1 + OFFSET2), local_sum1[0]);
        }
        __syncthreads();
    }

}

void ScalableVolumeRegistrationCudaKernelCaller::BuildLinearSystem(
    ScalableVolumeRegistrationCuda &registration) {

    int N = registration.source_.N_;
    const dim3 blocks(registration.source_active_subvolumes_);
    const dim3 threads(N, N, N);
    BuildLinearSystemKernel<<< blocks, threads >>> (*registration.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
}
}