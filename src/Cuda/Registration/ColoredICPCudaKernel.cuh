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
    TransformEstimationCudaForColoredICPDevice estimation,
    CorrespondenceSetCudaDevice corres) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= corres.indices_.size()) return;

    int i = corres.indices_[idx];

    PointCloudCudaDevice &pcl = estimation.target_;
    Vector3f &vt = pcl.points()[i];
    Vector3f &nt = pcl.normals()[i];
    Vector3f &color = pcl.colors()[i];
    float it = (color(0) + color(1) + color(2)) / 3.0f;

    Matrix3f AtA(0);
    Vector3f Atb(0);

    int nn = 0, max_nn = corres.matrix_.max_rows_;
    for (int j = 1; j < max_nn; ++j) {
        int adj_idx = corres.matrix_(j, i);
        if (adj_idx == -1) break;

        Vector3f &vt_adj = pcl.points()[adj_idx];
        Vector3f vt_proj = vt_adj - (vt_adj - vt).dot(nt) * nt;
        Vector3f &color_adj = pcl.colors()[adj_idx];
        float it_adj = (color_adj(0) + color_adj(1) + color_adj(2)) / 3.0f;

        float a0 = vt_proj(0) - vt(0);
        float a1 = vt_proj(1) - vt(1);
        float a2 = vt_proj(2) - vt(2);
        float b = it_adj - it;

        AtA(0, 0) += a0 * a0;
        AtA(0, 1) += a0 * a1;
        AtA(0, 2) += a0 * a2;
        AtA(1, 1) += a1 * a1;
        AtA(1, 2) += a1 * a2;
        AtA(2, 2) += a2 * a2;
        Atb(0) += a0 * b;
        Atb(1) += a1 * b;
        Atb(2) += a2 * b;

        ++nn;
    }

    /* orthogonal constraint */
    float nn2 = nn * nn;
    AtA(0, 0) += nn2 * nt(0) * nt(0);
    AtA(0, 1) += nn2 * nt(0) * nt(1);
    AtA(0, 2) += nn2 * nt(0) * nt(2);
    AtA(1, 1) += nn2 * nt(1) * nt(1);
    AtA(1, 2) += nn2 * nt(1) * nt(2);
    AtA(2, 2) += nn2 * nt(2) * nt(2);

    /* Symmetry */
    AtA(1, 0) = AtA(0, 1);
    AtA(2, 0) = AtA(0, 2);
    AtA(2, 1) = AtA(1, 2);

    estimation.target_color_gradient_[i] = AtA.ldlt().Solve(Atb);
}

void TransformEstimationCudaForColoredICPKernelCaller
::ComputeColorGradeintKernelCaller(
    TransformEstimationCudaForColoredICP &estimation,
    CorrespondenceSetCuda &corres_for_color_gradient) {

    const dim3 blocks(
        DIV_CEILING(estimation.target_.points().size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    ComputeColorGradientKernel << < blocks, threads >> > (
        *estimation.server_, *corres_for_color_gradient.server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void ComputeResultsAndTransformationKernel(
    TransformEstimationCudaForColoredICPDevice estimation) {
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
ComputeResultsAndTransformationKernelCaller(
    TransformEstimationCudaForColoredICP &estimation) {

    const dim3 blocks(DIV_CEILING(estimation.correspondences_.indices_.size(),
                                  THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    ComputeResultsAndTransformationKernel<< < blocks, threads >> > (
        *estimation.server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d