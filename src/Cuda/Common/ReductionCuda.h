//
// Created by wei on 11/2/18.
//

#pragma once

#include "UtilsCuda.h"

namespace open3d {

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
#endif
}