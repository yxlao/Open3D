//
// Created by wei on 10/2/18.
//

#include "Reduction2DCuda.cuh"
#include <Cuda/Geometry/ImageCudaDevice.cuh>

namespace open3d {

template<typename VecType, typename T>
__global__
void ReduceSum2DKernel(ImageCudaServer<VecType> src, T *sum) {
    __shared__ T local_sum[THREAD_2D_UNIT * THREAD_2D_UNIT];

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    /** Proper initialization **/
    /** MUST guarantee this is 0, even if it is not in an image **/
    local_sum[tid] = 0;

    /** We are SAFE to return after initialization,
     * as long as the reduction strides are times of THREAD_2D_UNIT **/
    if (x >= src.width_ || y >= src.height_) return;

    for (int i = 0; i < TEST_ARRAY_SIZE; ++i) {
        __syncthreads();

        local_sum[tid] = T(src.at(x, y)(0));
        __syncthreads();

        BlockReduceSum<T>(local_sum, tid);
        if (tid == 0) atomicAdd(sum, local_sum[0]);
    }
}

template<typename T>
__device__
inline T blockReduceSumShuffle(T sum) {
    /** How many warps do we have? THREAD_2D_UNIT^2 / WAR_SIZE **/
    static __shared__ T warp_sum[THREAD_2D_UNIT * THREAD_2D_UNIT / WARP_SIZE];

    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;

    sum = WarpReduceSumShuffle<T>(sum);
    if (lane_id == 0) {
        warp_sum[warp_id] = sum;
    }
    __syncthreads();

    /**
     * Only fill in the first warp with values indexed by lane
     * (not that intuitive)
     **/
    sum = (thread_id < (THREAD_2D_UNIT * THREAD_2D_UNIT / WARP_SIZE)) ?
          warp_sum[lane_id] : 0;

    if (warp_id == 0) sum = WarpReduceSumShuffle<T>(sum);

    return sum;
}

template<typename VecType, typename T>
__global__
void ReduceSum2DShuffleKernel(ImageCudaServer<VecType> src, T *sum_total) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    for (int i = 0; i < TEST_ARRAY_SIZE; ++i) {
        T sum =
            (x >= src.width_ || y >= src.height_) ? 0 : T(src.at(x, y)(0));
        __syncthreads();
        sum = blockReduceSumShuffle(sum);
        if (threadIdx.x == 0) atomicAdd(sum_total, sum);
    }
}

/** Why is it so fast ??? **/
template<typename VecType, typename T>
__global__
void AtomicSumKernel(ImageCudaServer<VecType> src, T *sum_total) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= src.width_ || y >= src.height_) return;
    for (int i = 0; i < TEST_ARRAY_SIZE; ++i) {
        T sum = T(src.at(x, y)(0));
        atomicAdd(sum_total, sum);
    }
}
}