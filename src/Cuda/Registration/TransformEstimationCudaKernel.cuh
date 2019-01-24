//
// Created by wei on 1/17/19.
//

#include "TransformEstimationCuda.h"
#include <Cuda/Common/UtilsCuda.h>

namespace open3d {
namespace cuda {
__global__
void ComputeSumsKernel(
    TransformEstimationPointToPointCudaDevice estimation) {
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

    Vector3f &source = estimation.source_.points()[source_idx];
    Vector3f &target = estimation.target_.points()[target_idx];

    {
        const int OFFSET1 = 9;
        local_sum0[tid] = source(0);
        local_sum1[tid] = source(1);
        local_sum2[tid] = source(2);
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
            atomicAdd(&estimation.results_.at(0 + OFFSET1), local_sum0[0]);
            atomicAdd(&estimation.results_.at(1 + OFFSET1), local_sum1[0]);
            atomicAdd(&estimation.results_.at(2 + OFFSET1), local_sum2[0]);
        }
        __syncthreads();
    }

    {
        const int OFFSET2 = 12;
        local_sum0[tid] = target(0);
        local_sum1[tid] = target(1);
        local_sum2[tid] = target(2);
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
            atomicAdd(&estimation.results_.at(0 + OFFSET2), local_sum0[0]);
            atomicAdd(&estimation.results_.at(1 + OFFSET2), local_sum1[0]);
            atomicAdd(&estimation.results_.at(2 + OFFSET2), local_sum2[0]);
        }
        __syncthreads();
    }
}

__global__
void ComputeResultsAndTransformationKernel(
    TransformEstimationPointToPointCudaDevice estimation) {
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

    Matrix3f Sigma;
    float sigma_source2, rmse;
    estimation.ComputePointwiseStatistics(
        source_idx, target_idx, Sigma, sigma_source2, rmse);

    {
        for (size_t i = 0; i < 3; i ++) {
            local_sum0[tid] = Sigma(i, 0);
            local_sum1[tid] = Sigma(i, 1);
            local_sum2[tid] = Sigma(i, 2);
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
                atomicAdd(&estimation.results_.at(3 * i + 0),
                          local_sum0[0]);
                atomicAdd(&estimation.results_.at(3 * i + 1),
                          local_sum1[0]);
                atomicAdd(&estimation.results_.at(3 * i + 2),
                          local_sum2[0]);
            }
            __syncthreads();
        }
    }

    {
        const int OFFSET3 = 15;
        local_sum0[tid] = sigma_source2;
        local_sum1[tid] = rmse;
        __syncthreads();

        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
            local_sum1[tid] += local_sum1[tid + 128];
        }
        __syncthreads();

        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
            local_sum1[tid] += local_sum1[tid + 64];
        }
        __syncthreads();

        if (tid < 32) {
            WarpReduceSum<float>(local_sum0, tid);
            WarpReduceSum<float>(local_sum1, tid);
        }

        if (tid == 0) {
            atomicAdd(&estimation.results_.at(0 + OFFSET3), local_sum0[0]);
            atomicAdd(&estimation.results_.at(1 + OFFSET3), local_sum1[0]);
        }
        __syncthreads();
    }
}


void TransformEstimationPointToPointCudaKernelCaller::
ComputeSumsKernelCaller(TransformEstimationPointToPointCuda &estimation){

    const dim3 blocks(DIV_CEILING(estimation.correspondences_.indices_.size(),
                                  THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    ComputeSumsKernel << < blocks, threads >> > (*estimation.server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

void TransformEstimationPointToPointCudaKernelCaller::
ComputeResultsAndTransformationKernelCaller(
    TransformEstimationPointToPointCuda &estimation) {

    const dim3 blocks(DIV_CEILING(estimation.correspondences_.indices_.size(),
                                  THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    ComputeResultsAndTransformationKernel<< < blocks, threads >> > (*estimation.server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}


/***************************************/
/** PointToPlane **/
__global__
void ComputeResultsAndTransformationKernel(
    TransformEstimationPointToPlaneCudaDevice estimation) {
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

    Vector6f jacobian, Jtr;
    float residual;
    HessianCuda<6> JtJ;

    estimation.ComputePointwiseJacobianAndResidual(
        source_idx, target_idx, jacobian, residual);
    ComputeJtJAndJtr(jacobian, residual, JtJ, Jtr);

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
        local_sum0[tid] = residual * residual;
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

void TransformEstimationPointToPlaneCudaKernelCaller::
ComputeResultsAndTransformationKernelCaller(
    TransformEstimationPointToPlaneCuda &estimation) {

    const dim3 blocks(DIV_CEILING(estimation.correspondences_.indices_.size(),
                                  THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    ComputeResultsAndTransformationKernel<< < blocks, threads >> > (*estimation.server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

} // cuda
} // open3d