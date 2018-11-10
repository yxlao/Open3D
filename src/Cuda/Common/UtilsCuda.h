//
// Created by wei on 9/27/18.
//

#pragma once

#include <cstdlib>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "Common.h"
#include "HelperCuda.h"

namespace open3d {

/** If this is on, perform boundary checks! **/
#define CUDA_DEBUG_ENABLE_ASSERTION_
#define CUDA_DEBUG_ENABLE_PRINTF_
#define HOST_DEBUG_MONITOR_LIFECYCLE_

#define CheckCuda(val)  check ( (val), #val, __FILE__, __LINE__ )

#ifdef __CUDACC__
__device__
inline float atomicMinf(float *addr, float value) {
    float old;
    old = (value >= 0) ?
        __int_as_float(atomicMin((int *) addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int *) addr, __float_as_uint(value)));
    return old;
}

__device__
inline float atomicMaxf(float *addr, float value) {
    float old;
    old = (value >= 0) ?
        __int_as_float(atomicMax((int *) addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int *) addr, __float_as_uint(value)));
    return old;
}
#endif

/** Ensure it is only visible for nvcc **/
#ifdef __CUDACC__
/**
 * This is SIMD: each += operation is synchronized over 32 threads at exactly
 * the same time
 */
template<typename T>
__DEVICE__
inline void WarpReduceSum(volatile T *local_sum, const int tid) {
    local_sum[tid] += local_sum[tid + 32];
    local_sum[tid] += local_sum[tid + 16];
    local_sum[tid] += local_sum[tid + 8];
    local_sum[tid] += local_sum[tid + 4];
    local_sum[tid] += local_sum[tid + 2];
    local_sum[tid] += local_sum[tid + 1];
}

template<typename T>
__DEVICE__
inline void WarpReduceMin(volatile T *local_min, const int tid) {
    local_min[tid] = (local_min[tid] < local_min[tid + 32]) ?
                     local_min[tid] : local_min[tid + 32];
    local_min[tid] = (local_min[tid] < local_min[tid + 16]) ?
                     local_min[tid] : local_min[tid + 16];
    local_min[tid] = (local_min[tid] < local_min[tid + 8]) ?
                     local_min[tid] : local_min[tid + 8];
    local_min[tid] = (local_min[tid] < local_min[tid + 4]) ?
                     local_min[tid] : local_min[tid + 4];
    local_min[tid] = (local_min[tid] < local_min[tid + 2]) ?
                     local_min[tid] : local_min[tid + 2];
    local_min[tid] = (local_min[tid] < local_min[tid + 1]) ?
                     local_min[tid] : local_min[tid + 1];
}

template<typename T>
__DEVICE__
inline void WarpReduceMax(volatile T *local_max, const int tid) {
    local_max[tid] = (local_max[tid + 32] < local_max[tid]) ?
                     local_max[tid] : local_max[tid + 32];
    local_max[tid] = (local_max[tid + 16] < local_max[tid]) ?
                     local_max[tid] : local_max[tid + 16];
    local_max[tid] = (local_max[tid + 8] < local_max[tid]) ?
                     local_max[tid] : local_max[tid + 8];
    local_max[tid] = (local_max[tid + 4] < local_max[tid]) ?
                     local_max[tid] : local_max[tid + 4];
    local_max[tid] = (local_max[tid + 2] < local_max[tid]) ?
                     local_max[tid] : local_max[tid + 2];
    local_max[tid] = (local_max[tid + 1] < local_max[tid]) ?
                     local_max[tid] : local_max[tid + 1];
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
#endif
}

