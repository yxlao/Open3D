//
// Created by wei on 9/27/18.
//

#pragma once

#include <cstdlib>

#include <Open3D/Utility/Console.h>
#include <Open3D/Utility/Timer.h>

#include "driver_types.h"
#include "cuda_runtime_api.h"

#include "Common.h"
#include "HelperCuda.h"

namespace open3d {

namespace cuda {
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
/** SUM **/
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
inline void DoubleBlockReduceSum(
    volatile T *local_sum0, volatile T *local_sum1, const int tid) {
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
}

template<typename T>
__device__
inline void TripleBlockReduceSum(
    volatile T *local_sum0, volatile T *local_sum1, volatile T *local_sum2,
    const int tid) {
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

/** MIN **/
template<typename T>
__device__
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
__device__
inline void BlockReduceMin(volatile T *local_min, const int tid) {
    if (tid < 128) {
        local_min[tid] = (local_min[tid] < local_min[tid + 128]) ?
                         local_min[tid] : local_min[tid + 128];
    }
    __syncthreads();
    if (tid < 64) {
        local_min[tid] = (local_min[tid + 64] < local_min[tid]) ?
                         local_min[tid] : local_min[tid + 64];
    }
    __syncthreads();
    if (tid < 32) WarpReduceMin<T>(local_min, tid);
}

template<typename T>
__device__
inline void TripleBlockReduceMin(
    volatile T *local_min0, volatile T *local_min1, volatile T *local_min2,
    const int tid) {
    if (tid < 128) {
        local_min0[tid] = (local_min0[tid] < local_min0[tid + 128]) ?
                          local_min0[tid] : local_min0[tid + 128];
        local_min1[tid] = (local_min1[tid] < local_min1[tid + 128]) ?
                          local_min1[tid] : local_min1[tid + 128];
        local_min2[tid] = (local_min2[tid] < local_min2[tid + 128]) ?
                          local_min2[tid] : local_min2[tid + 128];
    }
    __syncthreads();
    if (tid < 64) {
        local_min0[tid] = (local_min0[tid] < local_min0[tid + 64]) ?
                          local_min0[tid] : local_min0[tid + 64];
        local_min1[tid] = (local_min1[tid] < local_min1[tid + 64]) ?
                          local_min1[tid] : local_min1[tid + 64];
        local_min2[tid] = (local_min2[tid] < local_min2[tid + 64]) ?
                          local_min2[tid] : local_min2[tid + 64];
    }
    __syncthreads();
    if (tid < 32) {
        WarpReduceMin<T>(local_min0, tid);
        WarpReduceMin<T>(local_min1, tid);
        WarpReduceMin<T>(local_min2, tid);
    }
}

/** MAX **/
template<typename T>
__device__
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
inline void BlockReduceMax(volatile T *local_max, const int tid) {
    if (tid < 128) {
        local_max[tid] = (local_max[tid + 128] < local_max[tid]) ?
                         local_max[tid] : local_max[tid + 128];
    }
    __syncthreads();
    if (tid < 64) {
        local_max[tid] = (local_max[tid + 64] < local_max[tid]) ?
                         local_max[tid] : local_max[tid + 64];
    }
    __syncthreads();
    if (tid < 32) WarpReduceMax<T>(local_max, tid);
}

template<typename T>
__device__
inline void TripleBlockReduceMax(
    volatile T *local_max0, volatile T *local_max1, volatile T *local_max2,
    const int tid) {
    if (tid < 128) {
        local_max0[tid] = (local_max0[tid + 128] < local_max0[tid]) ?
                          local_max0[tid] : local_max0[tid + 128];
        local_max1[tid] = (local_max1[tid + 128] < local_max1[tid]) ?
                          local_max1[tid] : local_max1[tid + 128];
        local_max2[tid] = (local_max2[tid + 128] < local_max2[tid]) ?
                          local_max2[tid] : local_max2[tid + 128];
    }
    __syncthreads();
    if (tid < 64) {
        local_max0[tid] = (local_max0[tid + 64] < local_max0[tid]) ?
                          local_max0[tid] : local_max0[tid + 64];
        local_max1[tid] = (local_max1[tid + 64] < local_max1[tid]) ?
                          local_max1[tid] : local_max1[tid + 64];
        local_max2[tid] = (local_max2[tid + 64] < local_max2[tid]) ?
                          local_max2[tid] : local_max2[tid + 64];
    }
    __syncthreads();
    if (tid < 32) {
        WarpReduceMax<T>(local_max0, tid);
        WarpReduceMax<T>(local_max1, tid);
        WarpReduceMax<T>(local_max2, tid);
    }
}

#endif
} // cuda
} // open3d