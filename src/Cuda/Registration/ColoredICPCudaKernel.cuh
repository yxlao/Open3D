//
// Created by wei on 1/15/19.
//

#pragma once

#include "ColoredICPCuda.h"
#include <Cuda/Common/UtilsCuda.h>
#include <Cuda/Container/ArrayCudaDevice.cuh>
#include <Cuda/Container/Array2DCudaDevice.cuh>

namespace open3d {
namespace cuda {

__global__
void ComputeColorGradientKernel(
    TransformEstimationForColoredICPCudaDevice estimation,
    CorrespondenceSetCudaDevice corres_for_gradient) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= corres_for_gradient.indices_.size()) return;

    estimation.ComputePointwiseGradient(idx, corres_for_gradient);
}

void TransformEstimationCudaForColoredICPKernelCaller::ComputeColorGradeint(
    TransformEstimationForColoredICPCuda &estimation,
    CorrespondenceSetCuda &corres_for_color_gradient) {

    const dim3 blocks(
        DIV_CEILING(estimation.target_.points_.size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    ComputeColorGradientKernel << < blocks, threads >> > (
        *estimation.device_, *corres_for_color_gradient.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void ComputeResultsAndTransformationKernel(
    TransformEstimationForColoredICPCudaDevice estimation) {
    __shared__ float local_sum0[THREAD_1D_UNIT];
    __shared__ float local_sum1[THREAD_1D_UNIT];
    __shared__ float local_sum2[THREAD_1D_UNIT];

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= estimation.correspondences_.indices_.size()) return;

    const int tid = threadIdx.x;

    /** Proper initialization **/
    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    int source_idx = estimation.correspondences_.indices_[idx];
    int target_idx = estimation.correspondences_.matrix_(0, source_idx);

    Vector6f jacobian_I, jacobian_G, Jtr;
    float residual_I, residual_G;
    HessianCuda<6> JtJ;

    estimation.ComputePointwiseJacobianAndResidual(
        source_idx, target_idx, jacobian_I, jacobian_G, residual_I, residual_G);
    ComputeJtJAndJtr(jacobian_I, jacobian_G, residual_I, residual_G, JtJ, Jtr);

    /** Reduce Sum JtJ **/
#pragma unroll 1
    for (size_t i = 0; i < 21; i += 3) {
        local_sum0[tid] = JtJ(i + 0);
        local_sum1[tid] = JtJ(i + 1);
        local_sum2[tid] = JtJ(i + 2);
        __syncthreads();

        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
            local_sum1[tid] += local_sum1[tid + 128];
            local_sum2[tid] += local_sum2[tid + 128];
        }
        __syncthreads();

        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
            local_sum1[tid] += local_sum1[tid + 64];
            local_sum2[tid] += local_sum2[tid + 64];
        }
        __syncthreads();

        if (tid < 32) {
            WarpReduceSum<float>(local_sum0, tid);
            WarpReduceSum<float>(local_sum1, tid);
            WarpReduceSum<float>(local_sum2, tid);
        }

        if (tid == 0) {
            atomicAdd(&estimation.results_.at(i + 0), local_sum0[0]);
            atomicAdd(&estimation.results_.at(i + 1), local_sum1[0]);
            atomicAdd(&estimation.results_.at(i + 2), local_sum2[0]);
        }
        __syncthreads();
    }

    /** Reduce Sum Jtr **/
    const int OFFSET1 = 21;
#pragma unroll 1
    for (size_t i = 0; i < 6; i += 3) {
        local_sum0[tid] = Jtr(i + 0);
        local_sum1[tid] = Jtr(i + 1);
        local_sum2[tid] = Jtr(i + 2);
        __syncthreads();

        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
            local_sum1[tid] += local_sum1[tid + 128];
            local_sum2[tid] += local_sum2[tid + 128];
        }
        __syncthreads();

        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
            local_sum1[tid] += local_sum1[tid + 64];
            local_sum2[tid] += local_sum2[tid + 64];
        }
        __syncthreads();

        if (tid < 32) {
            WarpReduceSum<float>(local_sum0, tid);
            WarpReduceSum<float>(local_sum1, tid);
            WarpReduceSum<float>(local_sum2, tid);
        }

        if (tid == 0) {
            atomicAdd(&estimation.results_.at(i + 0 + OFFSET1), local_sum0[0]);
            atomicAdd(&estimation.results_.at(i + 1 + OFFSET1), local_sum1[0]);
            atomicAdd(&estimation.results_.at(i + 2 + OFFSET1), local_sum2[0]);
        }
        __syncthreads();
    }

    /** Reduce Sum rmse **/
    const int OFFSET2 = 27;
    {
        local_sum0[tid] = residual_I * residual_I + residual_G * residual_G;
        __syncthreads();

        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
        }
        __syncthreads();

        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
        }
        __syncthreads();

        if (tid < 32) {
            WarpReduceSum<float>(local_sum0, tid);
        }

        if (tid == 0) {
            atomicAdd(&estimation.results_.at(0 + OFFSET2), local_sum0[0]);
        }
        __syncthreads();
    }
}

void TransformEstimationCudaForColoredICPKernelCaller::
ComputeResultsAndTransformation(
    TransformEstimationForColoredICPCuda &estimation) {

    const dim3 blocks(DIV_CEILING(estimation.correspondences_.indices_.size(),
                                  THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    ComputeResultsAndTransformationKernel<< < blocks, threads >> > (
        *estimation.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d