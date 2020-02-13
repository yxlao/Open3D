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

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Open3D/Core/AdvancedIndexing.h"
#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/Indexer.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"

static constexpr int64_t default_grid_size = 64;
static constexpr int64_t default_block_size = 256;

namespace cg = cooperative_groups;

namespace open3d {
namespace kernel {
namespace cuda_launcher {

template <typename T>
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

// Specialize for double to avoid unaligned memory access compile errors.
template <>
struct SharedMemory<double> {
    __device__ inline operator double*() {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double*() const {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};

// The kernel needs a minimum of 64 * sizeof(scalar_t) bytes of shared
// memory.
// - blockDim.x <= 32: allocate 64 * sizeof(scalar_t) bytes.
// - blockDim.x > 32 : allocate blockDim.x * sizeof(scalar_t) bytes.
template <typename scalar_t>
int64_t GetSMSize(int64_t grid_size, int64_t block_size) {
    return (block_size <= 32) ? 2 * block_size * sizeof(scalar_t)
                              : block_size * sizeof(scalar_t);
}

std::pair<int64_t, int64_t> GetGridSizeBlockSize(int64_t n) {
    static auto NextPow2 = [](int64_t x) -> int64_t {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    };

    // block_size = NextPow2(ceil(n / 2))
    int64_t block_size = NextPow2((n + 1) / 2);
    block_size = std::min(block_size, default_block_size);

    // grid_size = ceil(n / (block_size * 2))
    int64_t grid_size = (n + (block_size * 2 - 1)) / (block_size * 2);
    grid_size = std::min(grid_size, default_grid_size);

    return std::make_pair(grid_size, block_size);
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
void LaunchReductionKernelOneOutput(const Indexer& indexer,
                                    func_t element_kernel) {
    OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

    int64_t n = indexer.NumWorkloads();
    int64_t grid_size = 0;
    int64_t block_size = 0;
    std::tie(grid_size, block_size) = GetGridSizeBlockSize(n);
    utility::LogInfo("n={}, grid_size={}, block_size={}", n, grid_size,
                     block_size);

    // Allocate device temporary memory. d_odata and d_tdata are double buffers
    // for recursive reductions.
    scalar_t* d_odata = nullptr;  // Device output, grid_size elements
    scalar_t* d_tdata = nullptr;  // Device temp output, grid_size elements
    OPEN3D_CUDA_CHECK(
            cudaMalloc((void**)&d_odata, grid_size * sizeof(scalar_t)));
    OPEN3D_CUDA_CHECK(
            cudaMalloc((void**)&d_tdata, grid_size * sizeof(scalar_t)));

    ReductionKernelOneOutput<scalar_t>
            <<<grid_size, block_size,
               GetSMSize<scalar_t>(grid_size, block_size)>>>(indexer,
                                                             element_kernel);
}

}  // namespace cuda_launcher
}  // namespace kernel
}  // namespace open3d
