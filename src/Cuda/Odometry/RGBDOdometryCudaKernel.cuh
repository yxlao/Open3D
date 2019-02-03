//
// Created by wei on 10/1/18.
//

#include "RGBDOdometryCudaDevice.cuh"

namespace open3d {
namespace cuda {
template<size_t N>
__global__
void DoSingleIterationKernel(RGBDOdometryCudaDevice<N> odometry, size_t level) {
    /** Add more memory blocks if we have **/
    /** TODO: check this version vs 1 __shared__ array version **/
    __shared__ float local_sum0[THREAD_2D_UNIT * THREAD_2D_UNIT];
    __shared__ float local_sum1[THREAD_2D_UNIT * THREAD_2D_UNIT];
    __shared__ float local_sum2[THREAD_2D_UNIT * THREAD_2D_UNIT];

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    /** Proper initialization **/
    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    if (x >= odometry.source_[level].depth_.width_
        || y >= odometry.source_[level].depth_.height_)
        return;

    int x_target, y_target;
    float residual_I, residual_D;
    Vector3f X_source_on_target;
    bool mask = odometry.ComputePixelwiseCorrespondenceAndResidual(
        x, y, level, x_target, y_target,
        X_source_on_target, residual_I, residual_D);

    Vector6f jacobian_I, jacobian_D, Jtr;
    HessianCuda<6> JtJ;
    mask = mask && odometry.ComputePixelwiseJacobian(
        x_target, y_target, level, X_source_on_target,
        jacobian_I, jacobian_D);
    if (mask) {
        odometry.correspondences_.push_back(Vector4i(x, y, x_target, y_target));
        ComputeJtJAndJtr(jacobian_I, jacobian_D, residual_I, residual_D,
            JtJ, Jtr);
    }

    /** Reduce Sum JtJ -> 2ms **/
    for (size_t i = 0; i < 21; i += 3) {
        local_sum0[tid] = mask ? JtJ(i + 0) : 0;
        local_sum1[tid] = mask ? JtJ(i + 1) : 0;
        local_sum2[tid] = mask ? JtJ(i + 2) : 0;
        __syncthreads();

        TripleBlockReduceSum<float>(local_sum0, local_sum1, local_sum2, tid);

        if (tid == 0) {
            atomicAdd(&odometry.results_.at(i + 0), local_sum0[0]);
            atomicAdd(&odometry.results_.at(i + 1), local_sum1[0]);
            atomicAdd(&odometry.results_.at(i + 2), local_sum2[0]);
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

        TripleBlockReduceSum<float>(local_sum0, local_sum1, local_sum2, tid);

        if (tid == 0) {
            atomicAdd(&odometry.results_.at(i + 0 + OFFSET1), local_sum0[0]);
            atomicAdd(&odometry.results_.at(i + 1 + OFFSET1), local_sum1[0]);
            atomicAdd(&odometry.results_.at(i + 2 + OFFSET1), local_sum2[0]);
        }
        __syncthreads();
    }

    /** Reduce Sum loss and inlier **/
    const int OFFSET2 = 27;
    {
        local_sum0[tid] = mask ?
            residual_I * residual_I + residual_D * residual_D : 0;
        local_sum1[tid] = mask ? 1 : 0;
        __syncthreads();

        DoubleBlockReduceSum<float>(local_sum0, local_sum1, tid);

        if (tid == 0) {
            atomicAdd(&odometry.results_.at(0 + OFFSET2), local_sum0[0]);
            atomicAdd(&odometry.results_.at(1 + OFFSET2), local_sum1[0]);
        }
        __syncthreads();
    }
}

template<size_t N>
void RGBDOdometryCudaKernelCaller<N>::DoSingleIteration(
    RGBDOdometryCuda<N> &odometry, size_t level) {

    const dim3 blocks(
        DIV_CEILING(odometry.source_[level].depthf_.width_, THREAD_2D_UNIT),
        DIV_CEILING(odometry.source_[level].depthf_.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    DoSingleIterationKernel << < blocks, threads >> > (
        *odometry.device_, level);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<size_t N>
__global__
void ComputeInformationMatrixKernel(RGBDOdometryCudaDevice<N> odometry) {
    /** Add more memory blocks if we have **/
    /** TODO: check this version vs 1 __shared__ array version **/
    __shared__ float local_sum0[THREAD_2D_UNIT * THREAD_2D_UNIT];
    __shared__ float local_sum1[THREAD_2D_UNIT * THREAD_2D_UNIT];
    __shared__ float local_sum2[THREAD_2D_UNIT * THREAD_2D_UNIT];

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    /** Proper initialization **/
    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    if (x >= odometry.source_[0].depth_.width_
        || y >= odometry.source_[0].depth_.height_)
        return;

    Vector6f jacobian_x, jacobian_y, jacobian_z;
    HessianCuda<6> JtJ;
    bool mask = odometry.ComputePixelwiseCorrespondenceAndInformationJacobian(
        x, y, jacobian_x, jacobian_y, jacobian_z);

    if (mask) {
        ComputeJtJ(jacobian_x, jacobian_y, jacobian_z, JtJ);
    }

    /** Reduce Sum JtJ -> 2ms **/
    for (size_t i = 0; i < 21; i += 3) {
        local_sum0[tid] = mask ? JtJ(i + 0) : 0;
        local_sum1[tid] = mask ? JtJ(i + 1) : 0;
        local_sum2[tid] = mask ? JtJ(i + 2) : 0;
        __syncthreads();

        TripleBlockReduceSum<float>(local_sum0, local_sum1, local_sum2, tid);

        if (tid == 0) {
            atomicAdd(&odometry.results_.at(i + 0), local_sum0[0]);
            atomicAdd(&odometry.results_.at(i + 1), local_sum1[0]);
            atomicAdd(&odometry.results_.at(i + 2), local_sum2[0]);
        }
        __syncthreads();
    }
}

template<size_t N>
void RGBDOdometryCudaKernelCaller<N>::ComputeInformationMatrix(
    RGBDOdometryCuda<N> &odometry) {

    const dim3 blocks(
        DIV_CEILING(odometry.source_[0].depthf_.width_, THREAD_2D_UNIT),
        DIV_CEILING(odometry.source_[0].depthf_.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ComputeInformationMatrixKernel << < blocks, threads >> >(*odometry.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d