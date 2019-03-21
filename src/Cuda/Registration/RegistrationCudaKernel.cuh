//
// Created by wei on 3/19/19.
//

#pragma once

#include <Cuda/Common/UtilsCuda.h>
#include <Cuda/Common/JacobianCuda.h>
#include <Cuda/Common/ReductionCuda.h>
#include <Cuda/Container/ArrayCudaDevice.cuh>
#include <Cuda/Container/Array2DCudaDevice.cuh>

#include <RegistrationCuda.h>

namespace open3d {
namespace cuda {

__global__
void ComputeColorGradientKernel(
    RegistrationCudaDevice estimation,
    CorrespondenceSetCudaDevice corres_for_gradient) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= corres_for_gradient.indices_.size()) return;

    estimation.ComputePointwiseColorGradient(idx, corres_for_gradient);
}

void RegistrationCudaKernelCaller::ComputeColorGradeint(
    RegistrationCuda &estimation,
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
void BuildLinearSystemForColoredICPKernel(
    RegistrationCudaDevice estimation) {
    __shared__ float local_sum0[THREAD_1D_UNIT];
    __shared__ float local_sum1[THREAD_1D_UNIT];
    __shared__ float local_sum2[THREAD_1D_UNIT];

    /** Proper initialization **/
    const int tid = threadIdx.x;

    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= estimation.correspondences_.indices_.size()) return;

    int source_idx = estimation.correspondences_.indices_[idx];
    int target_idx = estimation.correspondences_.matrix_(0, source_idx);

    Vector6f jacobian_I, jacobian_G, Jtr;
    float residual_I, residual_G;
    HessianCuda<6> JtJ;

    estimation.ComputePointwiseColoredJacobianAndResidual(
        source_idx, target_idx, jacobian_I, jacobian_G, residual_I, residual_G);
    ComputeJtJAndJtr(jacobian_I, jacobian_G, residual_I, residual_G, JtJ, Jtr);

    /** Reduce Sum JtJ **/
    for (size_t i = 0; i < 21; i += 3) {
        local_sum0[tid] = JtJ(i + 0);
        local_sum1[tid] = JtJ(i + 1);
        local_sum2[tid] = JtJ(i + 2);
        __syncthreads();

        BlockReduceSum<float>(tid, local_sum0, local_sum1, local_sum2);

        if (tid == 0) {
            atomicAdd(&estimation.results_.at(i + 0), local_sum0[0]);
            atomicAdd(&estimation.results_.at(i + 1), local_sum1[0]);
            atomicAdd(&estimation.results_.at(i + 2), local_sum2[0]);
        }
        __syncthreads();
    }

    /** Reduce Sum Jtr **/
    const int OFFSET1 = 21;
    for (size_t i = 0; i < 6; i += 3) {
        local_sum0[tid] = Jtr(i + 0);
        local_sum1[tid] = Jtr(i + 1);
        local_sum2[tid] = Jtr(i + 2);
        __syncthreads();

        BlockReduceSum<float>(tid, local_sum0, local_sum1, local_sum2);

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

        BlockReduceSum<float>(tid, local_sum0);

        if (tid == 0) {
            atomicAdd(&estimation.results_.at(0 + OFFSET2), local_sum0[0]);
        }
        __syncthreads();
    }
}

void RegistrationCudaKernelCaller::BuildLinearSystemForColoredICP(
    RegistrationCuda &registration) {

    const dim3 blocks(DIV_CEILING(registration.correspondences_.indices_.size(),
                                  THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    BuildLinearSystemForColoredICPKernel << < blocks, threads >>
        > (
            *registration.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void BuildLinearSystemForPointToPlaneICPKernel(
    RegistrationCudaDevice estimation) {
    __shared__ float local_sum0[THREAD_1D_UNIT];
    __shared__ float local_sum1[THREAD_1D_UNIT];
    __shared__ float local_sum2[THREAD_1D_UNIT];

    const int tid = threadIdx.x;

    /** Proper initialization **/
    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= estimation.correspondences_.indices_.size()) return;

    int source_idx = estimation.correspondences_.indices_[idx];
    int target_idx = estimation.correspondences_.matrix_(0, source_idx);

    Vector6f jacobian, Jtr;
    float residual;
    HessianCuda<6> JtJ;

    estimation.ComputePointwisePointToPlaneJacobianAndResidual(
        source_idx, target_idx, jacobian, residual);
    ComputeJtJAndJtr(jacobian, residual, JtJ, Jtr);

    /** Reduce Sum JtJ **/
#pragma unroll 1
    for (size_t i = 0; i < 21; i += 3) {
        local_sum0[tid] = JtJ(i + 0);
        local_sum1[tid] = JtJ(i + 1);
        local_sum2[tid] = JtJ(i + 2);
        __syncthreads();

        BlockReduceSum<float>(tid, local_sum0, local_sum1, local_sum2);

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

        BlockReduceSum<float>(tid, local_sum0, local_sum1, local_sum2);

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

        BlockReduceSum<float>(tid, local_sum0);

        if (tid == 0) {
            atomicAdd(&estimation.results_.at(0 + OFFSET2), local_sum0[0]);
        }
        __syncthreads();
    }
}

void RegistrationCudaKernelCaller::BuildLinearSystemForPointToPlaneICP(
    RegistrationCuda &registration) {
    const dim3 blocks(DIV_CEILING(registration.correspondences_.indices_.size(),
                                  THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    BuildLinearSystemForPointToPlaneICPKernel << < blocks,
        threads >> > (
            *registration.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void ComputeSumForPointToPointICPKernel(
    RegistrationCudaDevice estimation) {
    __shared__ float local_sum0[THREAD_1D_UNIT];
    __shared__ float local_sum1[THREAD_1D_UNIT];
    __shared__ float local_sum2[THREAD_1D_UNIT];

    const int tid = threadIdx.x;

    /** Proper initialization **/
    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= estimation.correspondences_.indices_.size()) return;

    int source_idx = estimation.correspondences_.indices_[idx];
    int target_idx = estimation.correspondences_.matrix_(0, source_idx);

    Vector3f &source = estimation.source_.points_[source_idx];
    Vector3f &target = estimation.target_.points_[target_idx];

    const int OFFSET1 = 0;
    {
        local_sum0[tid] = source(0);
        local_sum1[tid] = source(1);
        local_sum2[tid] = source(2);
        __syncthreads();

        BlockReduceSum<float>(tid, local_sum0, local_sum1, local_sum2);

        if (tid == 0) {
            atomicAdd(&estimation.results_.at(0 + OFFSET1), local_sum0[0]);
            atomicAdd(&estimation.results_.at(1 + OFFSET1), local_sum1[0]);
            atomicAdd(&estimation.results_.at(2 + OFFSET1), local_sum2[0]);
        }
        __syncthreads();
    }

    const int OFFSET2 = 3;
    {
        local_sum0[tid] = target(0);
        local_sum1[tid] = target(1);
        local_sum2[tid] = target(2);
        __syncthreads();

        BlockReduceSum<float>(tid, local_sum0, local_sum1, local_sum2);

        if (tid == 0) {
            atomicAdd(&estimation.results_.at(0 + OFFSET2), local_sum0[0]);
            atomicAdd(&estimation.results_.at(1 + OFFSET2), local_sum1[0]);
            atomicAdd(&estimation.results_.at(2 + OFFSET2), local_sum2[0]);
        }
        __syncthreads();
    }
}

void RegistrationCudaKernelCaller::ComputeSumForPointToPointICP(
    RegistrationCuda &registration) {
    const dim3 blocks(DIV_CEILING(registration.correspondences_.indices_.size(),
                                  THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    ComputeSumForPointToPointICPKernel << < blocks, threads >> > (*registration.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void BuildLinearSystemForPointToPointICPKernel(
    RegistrationCudaDevice estimation,
    Vector3f mean_source, Vector3f mean_target) {
    __shared__ float local_sum0[THREAD_1D_UNIT];
    __shared__ float local_sum1[THREAD_1D_UNIT];
    __shared__ float local_sum2[THREAD_1D_UNIT];

    const int tid = threadIdx.x;

    /** Proper initialization **/
    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= estimation.correspondences_.indices_.size()) return;

    int source_idx = estimation.correspondences_.indices_[idx];
    int target_idx = estimation.correspondences_.matrix_(0, source_idx);

    Matrix3f Sigma;
    float sigma_source2, rmse;
    estimation.ComputePointwisePointToPointSigmaAndResidual(
        source_idx, target_idx, mean_source, mean_target,
        Sigma, sigma_source2, rmse);

    for (size_t i = 0; i < 3; i ++) {
        local_sum0[tid] = Sigma(i, 0);
        local_sum1[tid] = Sigma(i, 1);
        local_sum2[tid] = Sigma(i, 2);
        __syncthreads();

        BlockReduceSum<float>(tid, local_sum0, local_sum1, local_sum2);

        if (tid == 0) {
            atomicAdd(&estimation.results_.at(3 * i + 0), local_sum0[0]);
            atomicAdd(&estimation.results_.at(3 * i + 1), local_sum1[0]);
            atomicAdd(&estimation.results_.at(3 * i + 2), local_sum2[0]);
        }
        __syncthreads();
    }

    const int OFFSET3 = 9;
    {
        local_sum0[tid] = sigma_source2;
        local_sum1[tid] = rmse;
        __syncthreads();

        BlockReduceSum<float>(tid, local_sum0, local_sum1);

        if (tid == 0) {
            atomicAdd(&estimation.results_.at(0 + OFFSET3), local_sum0[0]);
            atomicAdd(&estimation.results_.at(1 + OFFSET3), local_sum1[0]);
        }
        __syncthreads();
    }
}

void RegistrationCudaKernelCaller::BuildLinearSystemForPointToPointICP(
    RegistrationCuda &registration,
    const Vector3f &mean_source, const Vector3f &mean_target) {

    const dim3 blocks(DIV_CEILING(registration.correspondences_.indices_.size(),
                                  THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    BuildLinearSystemForPointToPointICPKernel<< < blocks, threads >> > (
        *registration.device_, mean_source, mean_target);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void ComputeInformationMatrixKernel(RegistrationCudaDevice estimation) {
    __shared__ float local_sum0[THREAD_1D_UNIT];
    __shared__ float local_sum1[THREAD_1D_UNIT];
    __shared__ float local_sum2[THREAD_1D_UNIT];

    const int tid = threadIdx.x;

    /** Proper initialization **/
    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    CorrespondenceSetCudaDevice &corres = estimation.correspondences_;
    if (idx >= corres.indices_.size()) return;

    int source_idx = corres.indices_[idx];
    int target_idx = corres.matrix_(0, source_idx);

    Vector6f jacobian_x, jacobian_y, jacobian_z;
    HessianCuda<6> JtJ;

    Vector3f &point = estimation.target_.points_[target_idx];
    estimation.ComputePixelwiseInformationJacobian(point,
                                                   jacobian_x,
                                                   jacobian_y,
                                                   jacobian_z);
    ComputeJtJ(jacobian_x, jacobian_y, jacobian_z, JtJ);

    /** Reduce Sum JtJ **/
    for (size_t i = 0; i < 21; i += 3) {
        local_sum0[tid] = JtJ(i + 0);
        local_sum1[tid] = JtJ(i + 1);
        local_sum2[tid] = JtJ(i + 2);
        __syncthreads();

        BlockReduceSum<float>(tid, local_sum0, local_sum1, local_sum2);

        if (tid == 0) {
            atomicAdd(&estimation.results_.at(i + 0), local_sum0[0]);
            atomicAdd(&estimation.results_.at(i + 1), local_sum1[0]);
            atomicAdd(&estimation.results_.at(i + 2), local_sum2[0]);
        }
        __syncthreads();
    }
}

void RegistrationCudaKernelCaller::ComputeInformationMatrix(
    RegistrationCuda &estimation) {
    const dim3 blocks(DIV_CEILING(estimation.correspondences_.indices_.size(),
                                  THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    ComputeInformationMatrixKernel << < blocks, threads >> > (
        *estimation.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

} // cuda
} // open3d