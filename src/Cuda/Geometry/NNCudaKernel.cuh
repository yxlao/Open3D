//
// Created by wei on 1/21/19.
//

#pragma once

#include "NNCuda.h"
#include <Cuda/Container/Array2DCudaDevice.cuh>

namespace open3d {
namespace cuda {

/* Adapted from https://github.com/vincentfpgarcia/kNN-CUDA */
/**
 * Column-wise access PER THREAD is more cache friendly:
 * | thread 0 | thread 1 | thread 2 | ...
 *->  (0, 0)     (0, 1)     (0, 2)    parallel row-0 access
 *->  (1, 0)     (1, 1)     (1, 2)    parallel row-1 access
 *                ...
 **/
__global__
void ComputeDistancesKernel(NNCudaDevice nn) {
    __shared__ float shared_query[THREAD_2D_UNIT][THREAD_2D_UNIT];
    __shared__ float shared_ref[THREAD_2D_UNIT][THREAD_2D_UNIT];

    const int query_count = nn.query_.max_cols_;
    const int feature_size = nn.query_.max_rows_;
    const int ref_count = nn.ref_.max_cols_;

    const int tx = threadIdx.x, ty = threadIdx.y;

    int query_base = blockIdx.x * blockDim.x;
    int ref_base = blockIdx.y * blockDim.y;

    int query_idx = query_base + tx;
    int ref_idx_local = ref_base + tx;
    int ref_idx_global = ref_base + ty;

    bool mask_query = query_idx < query_count;
    bool mask_ref_local = ref_idx_local < ref_count;
    bool mask_ref_global = ref_idx_global < ref_count;

    float ssd = 0;
    for (int feature_batch = 0;
         feature_batch < feature_size;
         feature_batch += THREAD_2D_UNIT) {

        /* Here ty denotes feature idx */
        int feature_idx = feature_batch + ty;
        bool mask_feature = feature_idx < feature_size;

        shared_query[ty][tx] = (mask_query && mask_feature) ?
            nn.query_(feature_idx, query_idx) : 0;
        shared_ref[ty][tx] = (mask_ref_local && mask_feature) ?
            nn.ref_(feature_idx, ref_idx_local) : 0;
        __syncthreads();

        /* Here ty denotes reference entry index */
        if (mask_query && mask_ref_global) {
            for (int j = 0; j < THREAD_2D_UNIT; ++j) {
                float diff = shared_query[j][tx] - shared_ref[j][ty];
                ssd += diff * diff;
            }
        }
        __syncthreads();
    }

    if (mask_query && mask_ref_global) {
        nn.distance_matrix_.at(ref_idx_global, query_idx) = ssd;
    }
}

void NNCudaKernelCaller::ComputeDistancesKernelCaller(NNCuda &nn) {
    const dim3 blocks(DIV_CEILING(nn.query_.max_cols_, THREAD_2D_UNIT),
                      DIV_CEILING(nn.reference_.max_cols_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ComputeDistancesKernel<<<blocks, threads>>>(*nn.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}


__global__
void FindNNKernel(NNCudaDevice nn) {
    const int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ref_count = nn.ref_.max_cols_;
    if (query_idx >= nn.query_.max_cols_) return;

    int nn_idx = 0;
    float nn_dist = nn.distance_matrix_(0, query_idx);
    for (int i = 1; i < ref_count; ++i) {
        float dist = nn.distance_matrix_(i, query_idx);
        if (dist < nn_dist) {
            nn_dist = dist;
            nn_idx = i;
        }
    }

    nn.nn_idx_(0, query_idx) = nn_idx;
    nn.nn_dist_(0, query_idx) = nn_dist;
}

void NNCudaKernelCaller::FindNNKernelCaller(NNCuda &nn) {
    const dim3 blocks(DIV_CEILING(nn.query_.max_cols_, 256));
    const dim3 threads(256);

    FindNNKernel<<<blocks, threads>>>(*nn.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

} // cuda
} // open3d