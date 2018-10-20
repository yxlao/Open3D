//
// Created by wei on 10/3/18.
//

#pragma once

#include "Reduction2DCuda.h"

namespace open3d {

/**
 * This is SIMD: each += operation is synchronized over 32 threads at exactly
 * the same time
 */
template<typename T>
__device__
inline void WarpReduceSum(volatile T *local_sum, const int tid) {
    local_sum[tid] += local_sum[tid + 32];
    local_sum[tid] += local_sum[tid + 16];
    local_sum[tid] += local_sum[tid + 8];
    local_sum[tid] += local_sum[tid + 4];
    local_sum[tid] += local_sum[tid + 2];
    local_sum[tid] += local_sum[tid + 1];
}

template<typename T>
__device__
inline void BlockReduceSum(volatile T *local_sum, const int tid) {
    if (tid < 128) local_sum[tid] += local_sum[tid + 128];
    __syncthreads();
    if (tid < 64) local_sum[tid] += local_sum[tid + 64];
    __syncthreads();
    if (tid < 32) WarpReduceSum<T>(local_sum, tid);
}

template<typename T>
__device__
inline T WarpReduceSumShuffle(T &sum) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    return sum;
}

/** For testing **/
template<typename VecType, typename T>
T ReduceSum2D(ImageCuda<VecType> &src) {
    const dim3 blocks(DIV_CEILING(src.width(), THREAD_2D_UNIT),
                      DIV_CEILING(src.height(), THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);

    T *sum;
    CheckCuda(cudaMalloc(&sum, sizeof(T)));
    CheckCuda(cudaMemset(sum, 0, sizeof(T)));
    ReduceSum2DKernel << < blocks, threads >> > (*src.server(), sum);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    T ret;
    CheckCuda(cudaMemcpy(&ret, sum, sizeof(T), cudaMemcpyDeviceToHost));
    return ret;
}

template<typename VecType, typename T>
T ReduceSum2DShuffle(ImageCuda<VecType> &src) {
    const dim3 blocks(DIV_CEILING(src.width(), THREAD_2D_UNIT),
                      DIV_CEILING(src.height(), THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);

    T *sum;
    CheckCuda(cudaMalloc(&sum, sizeof(T)));
    CheckCuda(cudaMemset(sum, 0, sizeof(T)));
    ReduceSum2DShuffleKernel << < blocks, threads >> > (*src.server(), sum);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    T ret;
    CheckCuda(cudaMemcpy(&ret, sum, sizeof(T), cudaMemcpyDeviceToHost));
    return ret;
}

template<typename VecType, typename T>
T AtomicSum(ImageCuda<VecType> &src) {
    const dim3 blocks(DIV_CEILING(src.width(), THREAD_2D_UNIT),
                      DIV_CEILING(src.height(), THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);

    T *sum;
    CheckCuda(cudaMalloc(&sum, sizeof(T)));
    CheckCuda(cudaMemset(sum, 0, sizeof(T)));
    AtomicSumKernel << < blocks, threads >> > (*src.server(), sum);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    T ret;
    CheckCuda(cudaMemcpy(&ret, sum, sizeof(T), cudaMemcpyDeviceToHost));
    return ret;
}
}
