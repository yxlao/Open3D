//
// Created by wei on 1/21/19.
//

#pragma once

#include "NNCuda.h"
#include <Cuda/Container/Array2DCudaDevice.cuh>

namespace open3d {
namespace cuda {

/** Pass 1: block-wise
 * (query_rows, ref_rows / 256) x (1, 256)
 **/
__global__
void ComputeDistancesKernel(NNCudaDevice nn) {
    __shared__ float shared_query[THREAD_2D_UNIT][THREAD_2D_UNIT];
    __shared__ float shared_ref[THREAD_2D_UNIT][THREAD_2D_UNIT];

    const int query_max_row = nn.query_.max_rows_;
    const int feature_size = nn.query_.max_cols_;
    const int ref_max_row = nn.ref_.max_rows_;

    const int tx = threadIdx.x, ty = threadIdx.y;

    int query_base = blockIdx.x * blockDim.x;
    int ref_base = blockIdx.y * blockDim.y;

    int query_idx = query_base + tx;
    int ref_idx_local = ref_base + tx;
    int ref_idx_global = ref_base + ty;

    bool mask_query = query_idx < query_max_row;
    bool mask_ref_local = ref_idx_local < ref_max_row;
    bool mask_ref_global = ref_idx_global < ref_max_row;

    float ssd = 0;
    float *query = nn.query_.row(query_idx);
    float *ref = nn.ref_.row(ref_idx_local);

    for (int feature_batch = 0;
         feature_batch < feature_size;
         feature_batch += THREAD_2D_UNIT) {

        /* Here ty denotes feature idx */
        int feature_idx = feature_batch + ty;
        bool mask_feature = feature_idx < feature_size;

        shared_query[ty][tx] = (mask_query && mask_feature) ?
            query[feature_idx] : 0;
        shared_ref[ty][tx] = (mask_ref_local && mask_feature) ?
            ref[feature_idx] : 0;
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
        nn.distance_matrix_.at(query_idx, ref_idx_global) = ssd;
    }
}

void NNCudaKernelCaller::ComputeDistancesKernelCaller(NNCuda &nn) {
    const dim3 blocks(DIV_CEILING(nn.query_.max_rows_, THREAD_2D_UNIT),
                      DIV_CEILING(nn.reference_.max_rows_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ComputeDistancesKernel<<<blocks, threads>>>(*nn.server_);
    //CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}


/** (query_rows) x (256) **/
__global__
void FindNNKernel(NNCudaDevice nn) {
    const int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ref_count = nn.ref_.max_rows_;

    if (query_idx >= nn.query_.max_rows_) return;

    float *dists = nn.distance_matrix_.row(query_idx);

    int nn_idx = 0;
    float nn_dist = dists[0];
    for (int i = 1; i < ref_count; ++i) {
        float dist = dists[i];
        if (dist < nn_dist) {
            nn_dist = dist;
            nn_idx = i;
        }
    }

    nn.nn_idx_(query_idx, 0) = nn_idx;
    nn.nn_dist_(query_idx, 0) = nn_dist;
}

void NNCudaKernelCaller::FindNNKernelCaller(NNCuda &nn) {
    const dim3 blocks(DIV_CEILING(nn.query_.max_rows_, 256));
    const dim3 threads(256);

    FindNNKernel<<<blocks, threads>>>(*nn.server_);
    //CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

} // cuda
} // open3d