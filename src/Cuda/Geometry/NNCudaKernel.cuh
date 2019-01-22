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
void ComputeAndReduceDistancesKernel(NNCudaDevice nn) {
    const int query_idx = blockIdx.x;
    const int ref_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (ref_idx >= nn.ref_.max_rows_)
        return;

    /** Compute distance **/
    float ssd = 0;
    const int feature_size = nn.query_.max_cols_;
    float *query = &nn.query_(query_idx, 0);
    float *ref = &nn.ref_(ref_idx, 0);
    for (int i = 0; i < feature_size; ++i) {
        float diff = nn.query_(query_idx, i) - nn.ref_(ref_idx, i);
        ssd += diff * diff;
    }
    nn.distance_matrix_(query_idx, ref_idx) = ssd;
    __syncthreads();

    /** Pure loop
     *  We can use reduction with global memory,
     *  but it eats twice the memory **/
    if (threadIdx.y == 0) {
        int nn_idx = ref_idx;
        float nn_dist = ssd;

        for (int i = 1; i < 256; ++i) {
            int ref_idx_i = ref_idx + i;
            if (ref_idx_i >= nn.ref_.max_rows_) break;
            const float dist_i = nn.distance_matrix_(query_idx, ref_idx_i);
            if (dist_i < nn_dist) {
                nn_dist = dist_i;
                nn_idx = ref_idx_i;
            }
        }

        nn.nn_dist_(query_idx, blockIdx.y) = nn_dist;
        nn.nn_idx_(query_idx, blockIdx.y) = nn_idx;
    }
}

/** (query_rows) x (256) **/
__global__
void ReduceBlockwiseDistancesKernel(NNCudaDevice nn, int ref_blocks) {
    __shared__ float nn_dist[256];
    __shared__ int nn_idx[256];

    const int query_idx = blockIdx.x;
    const int tid = threadIdx.x;

    nn_dist[tid] = 1e10;
    nn_idx[tid] = 0;

    if (tid >= ref_blocks) return;

    nn_dist[tid] = nn.nn_dist_(query_idx, tid);
    nn_idx[tid] = nn.nn_idx_(query_idx, tid);
    __syncthreads();

    /** Local reduction **/
#pragma unroll 1
    for (int shift = 7; shift >= 0; --shift) {
        int offset = (1 << shift);
        if (nn_dist[tid + offset] < nn_dist[tid]) {
            nn_dist[tid] = nn_dist[tid + offset];
            nn_idx[tid] = nn_idx[tid + offset];
        }
        /* If operations are not synchronized in a warp */
        __syncthreads();
    }

    if (tid == 0) {
        nn.nn_dist_(query_idx, 0) = nn_dist[0];
        nn.nn_idx_(query_idx, 0) = nn_idx[0];
    }
}

void NNCudaKernelCaller::ComputeAndReduceDistancesKernelCaller(NNCuda &nn) {
    /** reference_.max_rows_ should not > 256 x 256 = 65536.
      * Otherwise the second reduction will not be runnable **/
    assert(nn.reference_.max_rows_ <= 65536);

    nn.ref_blocks_ = DIV_CEILING(nn.reference_.max_rows_, 256);
    const dim3 blocks(nn.query_.max_rows_, nn.ref_blocks_);
    const dim3 threads(1, 256);
    ComputeAndReduceDistancesKernel<<<blocks, threads>>>(*nn.server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

void NNCudaKernelCaller::ReduceBlockwiseDistancesKernelCaller(NNCuda &nn) {
    const dim3 blocks(nn.query_.max_rows_);
    const dim3 threads(256);

    ReduceBlockwiseDistancesKernel<<<blocks, threads>>>(*nn.server_, nn.ref_blocks_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

} // cuda
} // open3d