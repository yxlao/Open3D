//
// Created by wei on 10/2/18.
//

#pragma once

#include <src/Cuda/Common/UtilsCuda.h>
#include <src/Cuda/Geometry/ImageCuda.h>
#include <src/Cuda/Common/LinearAlgebraCuda.h>

/**
 * Reduction is a PER-BLOCK operation.
 * Typically, for a 16x16 block, we get the accumulated value;
 * then, we atomicAdd them into a global memory to get the sum.
 * https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
*/

namespace open3d {
namespace cuda {

#define WARP_SIZE 32

template<typename T>
__DEVICE__
inline void BlockReduceSum(volatile T *local_sum, int tid);

template<typename T>
__DEVICE__
inline T WarpReduceSumShuffle(T &sum);

/** The rest are for testing **/
#define TEST_ARRAY_SIZE (6 + 21 + 2)

template<typename Scalar, size_t Channel>
__GLOBAL__
void ReduceSum2DKernel(ImageCudaDevice<Scalar, Channel> src, Scalar *sum);

template<typename Scalar, size_t Channel>
__GLOBAL__
void ReduceSum2DShuffleKernel(ImageCudaDevice<Scalar, Channel> src, Scalar*sum);

template<typename Scalar, size_t Channel>
__GLOBAL__
void AtomicSumKernel(ImageCudaDevice<Scalar, Channel> src, Scalar *sum);

template<typename Scalar, size_t Channel>
Scalar ReduceSum2D(ImageCuda<Scalar, Channel> &src);

template<typename Scalar, size_t Channel>
Scalar ReduceSum2DShuffle(ImageCuda<Scalar, Channel> &src);

template<typename Scalar, size_t Channel>
Scalar AtomicSum(ImageCuda<Scalar, Channel> &src);
} // cuda
} // open3d