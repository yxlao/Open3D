//
// Created by wei on 3/20/19.
//

#pragma once

#include "Common.h"
namespace open3d {
namespace cuda {

/** Ensure it is only visible for nvcc **/

/** SUM **/
/**
 * This is SIMD: each += operation is synchronized over 32 threads at exactly
 * the same time
 */

#ifndef __CUDACC__
#define __syncthreads()
#endif

#ifdef __CUDACC__
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

/** SUM **/
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

template<typename T, size_t BLOCK_SIZE>
__DEVICE__
inline void BlockReduceSum(
    const int tid, volatile T *local_sum) {
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) { local_sum[tid] += local_sum[tid + 256]; }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) { local_sum[tid] += local_sum[tid + 128]; }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) { local_sum[tid] += local_sum[tid + 64]; }
        __syncthreads();
    }
    if (tid < 32) { WarpReduceSum<T>(local_sum, tid); }
}

template<typename T, size_t BLOCK_SIZE>
__DEVICE__
inline void BlockReduceSum(
    const int tid, volatile T *local_sum0, volatile T *local_sum1) {
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            local_sum0[tid] += local_sum0[tid + 256];
            local_sum1[tid] += local_sum1[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
            local_sum1[tid] += local_sum1[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
            local_sum1[tid] += local_sum1[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduceSum<float>(local_sum0, tid);
        WarpReduceSum<float>(local_sum1, tid);
    }
}

template<typename T, size_t BLOCK_SIZE>
__DEVICE__
inline void BlockReduceSum(
    const int tid,
    volatile T *local_sum0, volatile T *local_sum1, volatile T *local_sum2) {
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            local_sum0[tid] += local_sum0[tid + 256];
            local_sum1[tid] += local_sum1[tid + 256];
            local_sum2[tid] += local_sum2[tid + 256];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
            local_sum1[tid] += local_sum1[tid + 128];
            local_sum2[tid] += local_sum2[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
            local_sum1[tid] += local_sum1[tid + 64];
            local_sum2[tid] += local_sum2[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduceSum<float>(local_sum0, tid);
        WarpReduceSum<float>(local_sum1, tid);
        WarpReduceSum<float>(local_sum2, tid);
    }
}

/** MIN **/
template<typename T>
__DEVICE__
inline void WarpReduceMin(const int tid, volatile T *local_min) {
    local_min[tid] = O3D_MIN(local_min[tid], local_min[tid + 32]);
    local_min[tid] = O3D_MIN(local_min[tid], local_min[tid + 16]);
    local_min[tid] = O3D_MIN(local_min[tid], local_min[tid + 8]);
    local_min[tid] = O3D_MIN(local_min[tid], local_min[tid + 4]);
    local_min[tid] = O3D_MIN(local_min[tid], local_min[tid + 2]);
    local_min[tid] = O3D_MIN(local_min[tid], local_min[tid + 1]);
}

template<typename T>
__DEVICE__
inline void BlockReduceMin(volatile T *local_min, const int tid) {
    if (tid < 128) { local_min[tid] = O3D_MIN(local_min[tid], local_min[tid] + 128); }
    __syncthreads();
    if (tid < 64) { local_min[tid] = O3D_MIN(local_min[tid], local_min[tid] + 64); }
    __syncthreads();
    if (tid < 32) { WarpReduceMin<T>(tid, local_min); }
}

template<typename T>
__DEVICE__
inline void BlockReduceMin(
    const int tid,
    volatile T *local_min0, volatile T *local_min1, volatile T *local_min2) {

    if (tid < 128) {
        local_min0[tid] = O3D_MIN(local_min0[tid], local_min0[tid + 128]);
        local_min1[tid] = O3D_MIN(local_min1[tid], local_min1[tid + 128]);
        local_min2[tid] = O3D_MIN(local_min2[tid], local_min2[tid + 128]);
    }
    __syncthreads();
    if (tid < 64) {
        local_min0[tid] = O3D_MIN(local_min0[tid], local_min0[tid + 64]);
        local_min1[tid] = O3D_MIN(local_min1[tid], local_min1[tid + 64]);
        local_min2[tid] = O3D_MIN(local_min2[tid], local_min2[tid + 64]);
    }
    __syncthreads();
    if (tid < 32) {
        WarpReduceMin<T>(tid, local_min0);
        WarpReduceMin<T>(tid, local_min1);
        WarpReduceMin<T>(tid, local_min2);
    }
}

/** MAX **/
template<typename T>
__DEVICE__
inline void WarpReduceMax(const int tid, volatile T *local_max) {
    local_max[tid] = O3D_MAX(local_max[tid], local_max[tid + 32]);
    local_max[tid] = O3D_MAX(local_max[tid], local_max[tid + 16]);
    local_max[tid] = O3D_MAX(local_max[tid], local_max[tid + 8]);
    local_max[tid] = O3D_MAX(local_max[tid], local_max[tid + 4]);
    local_max[tid] = O3D_MAX(local_max[tid], local_max[tid + 2]);
    local_max[tid] = O3D_MAX(local_max[tid], local_max[tid + 1]);
}

template<typename T>
__DEVICE__
inline void BlockReduceMax(const int tid, volatile T *local_max) {
    if (tid < 128) { local_max[tid] = O3D_MAX(local_max[tid], local_max[tid + 128]); }
    __syncthreads();
    if (tid < 64) { local_max[tid] = O3D_MAX(local_max[tid], local_max[tid + 64]); }
    __syncthreads();
    if (tid < 32) { WarpReduceMax<T>(tid, local_max); }
}

template<typename T>
__DEVICE__
inline void BlockReduceMax(
    const int tid,
    volatile T *local_max0, volatile T *local_max1, volatile T *local_max2) {
    if (tid < 128) {
        local_max0[tid] = O3D_MAX(local_max0[tid], local_max0[tid + 128]);
        local_max1[tid] = O3D_MAX(local_max1[tid], local_max1[tid + 128]);
        local_max2[tid] = O3D_MAX(local_max2[tid], local_max2[tid + 128]);
    }
    __syncthreads();
    if (tid < 64) {
        local_max0[tid] = O3D_MAX(local_max0[tid], local_max0[tid + 64]);
        local_max1[tid] = O3D_MAX(local_max1[tid], local_max1[tid + 64]);
        local_max2[tid] = O3D_MAX(local_max2[tid], local_max2[tid + 64]);
    }
    __syncthreads();
    if (tid < 32) {
        WarpReduceMax<T>(tid, local_max0);
        WarpReduceMax<T>(tid, local_max1);
        WarpReduceMax<T>(tid, local_max2);
    }
}
} // cuda
} // open3d