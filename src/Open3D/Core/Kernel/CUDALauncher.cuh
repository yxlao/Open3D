// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "Open3D/Core/AdvancedIndexing.h"
#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/Indexer.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"

// CUDA kernel launcher's goal is to separate scheduling (looping through each
// valid element) and computation (operations performed on each element).
//
// The kernel launch mechanism is inspired by PyTorch's launch Loops.cuh.
// See: https://tinyurl.com/y4lak257

static constexpr int64_t default_block_size = 128;
static constexpr int64_t default_thread_size = 4;

namespace open3d {
namespace kernel {
namespace cuda_launcher {

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
    __device__ inline operator T*() {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// Applies f for each element
// Works for unary / binary elementwise operations
template <int64_t block_size, int64_t thread_size, typename func_t>
__global__ void ElementWiseKernel(int64_t n, func_t f) {
    int64_t items_per_block = block_size * thread_size;
    int64_t idx = blockIdx.x * items_per_block + threadIdx.x;
#pragma unroll
    for (int64_t i = 0; i < thread_size; i++) {
        if (idx < n) {
            f(idx);
            idx += block_size;
        }
    }
}

template <typename scalar_t, typename func_t>
__global__ void ReductionKernelOneOutput(Indexer indexer,
                                         func_t element_kernel) {
    scalar_t* sdata = SharedMemory<scalar_t>();
    int64_t n = indexer.NumWorkloads();
    int64_t tid = threadIdx.x;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        sdata[tid] = *reinterpret_cast<scalar_t*>(indexer.GetInputPtr(0, i));
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    for (int64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < n) {
            // element_kernel(src, dst) -> dst = dst + src
            element_kernel(&sdata[tid + s], &sdata[tid]);
        }
        __syncthreads();
    }

    if (i < n && tid == 0) {
        *reinterpret_cast<scalar_t*>(indexer.GetOutputPtr(i)) = sdata[tid];
    }
}

template <typename scalar_t, typename func_t>
void LaunchUnaryEWKernel(const Indexer& indexer, func_t element_kernel) {
    OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

    int64_t n = indexer.NumWorkloads();
    int64_t items_per_block = default_block_size * default_thread_size;
    int64_t grid_size = (n + items_per_block - 1) / items_per_block;

    auto f = [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
        element_kernel(indexer.GetInputPtr(0, workload_idx),
                       indexer.GetOutputPtr(workload_idx));
    };

    ElementWiseKernel<default_block_size, default_thread_size>
            <<<grid_size, default_block_size, 0>>>(n, f);
}

template <typename scalar_t, typename func_t>
void LaunchBinaryEWKernel(const Indexer& indexer, func_t element_kernel) {
    OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

    int64_t n = indexer.NumWorkloads();
    int64_t items_per_block = default_block_size * default_thread_size;
    int64_t grid_size = (n + items_per_block - 1) / items_per_block;

    auto f = [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
        element_kernel(indexer.GetInputPtr(0, workload_idx),
                       indexer.GetInputPtr(1, workload_idx),
                       indexer.GetOutputPtr(workload_idx));
    };

    ElementWiseKernel<default_block_size, default_thread_size>
            <<<grid_size, default_block_size, 0>>>(n, f);
}

template <typename scalar_t, typename func_t>
void LaunchReductionKernelOneOutput(const Indexer& indexer,
                                    func_t element_kernel) {
    OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

    int64_t grid_size = 4;
    int64_t block_size = 128;

    ReductionKernelOneOutput<scalar_t>
            <<<grid_size, block_size, block_size * sizeof(scalar_t)>>>(
                    indexer, element_kernel);
}

template <typename scalar_t, typename func_t>
void LaunchAdvancedIndexerKernel(const AdvancedIndexer& indexer,
                                 func_t element_kernel) {
    OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

    int64_t n = indexer.NumWorkloads();
    int64_t items_per_block = default_block_size * default_thread_size;
    int64_t grid_size = (n + items_per_block - 1) / items_per_block;

    auto f = [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
        element_kernel(indexer.GetInputPtr(workload_idx),
                       indexer.GetOutputPtr(workload_idx));
    };

    ElementWiseKernel<default_block_size, default_thread_size>
            <<<grid_size, default_block_size, 0>>>(n, f);
}

}  // namespace cuda_launcher
}  // namespace kernel
}  // namespace open3d
