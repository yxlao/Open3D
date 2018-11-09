//
// Created by wei on 10/2/18.
//

#pragma once

#include <Cuda/Common/UtilsCuda.h>
#include <Cuda/Geometry/ImageCuda.h>
#include <Cuda/Common/VectorCuda.h>

/**
 * Reduction is a PER-BLOCK operation.
 * Typically, for a 16x16 block, we get the accumulated value;
 * then, we atomicAdd them into a global memory to get the sum.
 * https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
*/

namespace open3d {

#define WARP_SIZE 32

template<typename T>
__DEVICE__
inline void BlockReduceSum(volatile T *local_sum, int tid);

template<typename T>
__DEVICE__
inline T WarpReduceSumShuffle(T &sum);

/** The rest are for testing **/
#define TEST_ARRAY_SIZE (6 + 21 + 2)

template<typename VecType, typename T>
__GLOBAL__
void ReduceSum2DKernel(ImageCudaServer<VecType> src, T *sum);

template<typename VecType, typename T>
__GLOBAL__
void ReduceSum2DShuffleKernel(ImageCudaServer<VecType> src, T *sum);

template<typename VecType, typename T>
__GLOBAL__
void AtomicSumKernel(ImageCudaServer<VecType> src, T *sum);

template<typename VecType, typename T>
T ReduceSum2D(ImageCuda<VecType> &src);

template<typename VecType, typename T>
T ReduceSum2DShuffle(ImageCuda<VecType> &src);

template<typename VecType, typename T>
T AtomicSum(ImageCuda<VecType> &src);
}
