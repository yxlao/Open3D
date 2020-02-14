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
__global__ void ReduceKernelInit(Indexer indexer,
                                 scalar_t identity,
                                 func_t element_kernel,
                                 scalar_t* g_odata,
                                 unsigned int n) {
    cg::thread_block cta = cg::this_thread_block();
    scalar_t* sdata = SharedMemory<scalar_t>();
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int grid_stride = blockDim.x * 2 * gridDim.x;

    // Reduce multiple elements per thread. Larger gridDim.x results in larger
    // grid_stride and fewer elements per thread.
    scalar_t local_result = identity;
    while (i < n) {
        // local_result += g_idata[i];
        element_kernel(indexer.GetInputPtr(0, i), &local_result);
        if (i + blockDim.x < n) {
            // local_result += g_idata[i + blockDim.x];
            element_kernel(indexer.GetInputPtr(0, i + blockDim.x),
                           &local_result);
        }
        i += grid_stride;
    }
    sdata[tid] = local_result;
    cg::sync(cta);

    // Unrolled: 512, 256, 128.
    if (blockDim.x >= 512 && tid < 256) {
        // local_result += sdata[tid + 256];
        element_kernel(indexer.GetInputPtr(0, tid + 256), &local_result);
        sdata[tid] = local_result;
    }
    cg::sync(cta);
    if (blockDim.x >= 256 && tid < 128) {
        // local_result += sdata[tid + 128];
        element_kernel(indexer.GetInputPtr(0, tid + 128), &local_result);
        sdata[tid] = local_result;
    }
    cg::sync(cta);
    if (blockDim.x >= 128 && tid < 64) {
        // local_result += sdata[tid + 64];
        element_kernel(indexer.GetInputPtr(0, tid + 64), &local_result);
        sdata[tid] = local_result;
    }
    cg::sync(cta);

    // // Last 2nd warp
    // scalar_t local_temp;
    // if (blockDim.x >= 64 && tid < 32) {
    //     local_temp = sdata[tid + 32];
    //     element_kernel(&local_temp, &local_result);
    //     sdata[tid] = local_result;
    // }
    // if (tid < 32) {
    //     printf("After offset %d, sdata[%d]=%f\n", 32, tid,
    //            static_cast<float>(sdata[tid]));
    // }

    // // Last warp
    // cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    // if (blockDim.x >= 32) {
    //     local_temp = tile32.shfl_down(local_result, 16);
    //     element_kernel(&local_temp, &local_result);
    //     sdata[tid] = local_result;
    // }
    // if (tid < 16) {
    //     printf("After offset %d, sdata[%d]=%f, local_temp=%f, "
    //            "local_result=%f\n",
    //            16, tid, static_cast<float>(sdata[tid]),
    //            static_cast<float>(local_temp),
    //            static_cast<float>(local_result));
    // }

    // if (blockDim.x >= 16) {
    //     local_temp = tile32.shfl_down(local_result, 8);
    //     element_kernel(&local_temp, &local_result);
    //     sdata[tid] = local_result;
    // }
    // if (tid < 8) {
    //     printf("After offset %d, sdata[%d]=%f, local_temp=%f, "
    //            "local_result=%f\n",
    //            8, tid, static_cast<float>(sdata[tid]),
    //            static_cast<float>(local_temp),
    //            static_cast<float>(local_result));
    // }

    // if (blockDim.x >= 8) {
    //     local_temp = tile32.shfl_down(local_result, 4);
    //     element_kernel(&local_temp, &local_result);
    //     sdata[tid] = local_result;
    // }
    // if (tid < 4) {
    //     printf("After offset %d, sdata[%d]=%f, local_temp=%f, "
    //            "local_result=%f\n",
    //            4, tid, static_cast<float>(sdata[tid]),
    //            static_cast<float>(local_temp),
    //            static_cast<float>(local_result));
    // }

    // if (blockDim.x >= 4) {
    //     local_temp = tile32.shfl_down(local_result, 2);
    //     element_kernel(&local_temp, &local_result);
    //     sdata[tid] = local_result;
    // }
    // if (tid < 2) {
    //     printf("After offset %d, sdata[%d]=%f, local_temp=%f, "
    //            "local_result=%f\n",
    //            2, tid, static_cast<float>(sdata[tid]),
    //            static_cast<float>(local_temp),
    //            static_cast<float>(local_result));
    // }

    // if (blockDim.x >= 2) {
    //     local_temp = tile32.shfl_down(local_result, 1);
    //     element_kernel(&local_temp, &local_result);
    //     sdata[tid] = local_result;
    // }
    // if (tid < 1) {
    //     printf("After offset %d, sdata[%d]=%f, local_temp=%f, "
    //            "local_result=%f\n",
    //            1, tid, static_cast<float>(sdata[tid]),
    //            static_cast<float>(local_temp),
    //            static_cast<float>(local_result));
    // }

    // Single warp reduction with shuffle: 64, 32, 16, 8, 4, 2, 1
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    scalar_t local_temp = identity;
    if (cta.thread_rank() < 32) {
        // Fetch final intermediate result from 2nd warp
        if (blockDim.x >= 64) {
            // local_result += sdata[tid + 32];
            element_kernel(indexer.GetInputPtr(0, tid + 32), &local_result);
        }
        // Reduce final warp using shuffle
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            if (blockDim.x >= offset * 2) {
                scalar_t original = local_result;
                local_temp = tile32.shfl_down(local_result, offset);
                element_kernel(&local_temp, &local_result);
                printf("rank %d, offset %d, original %f, local_temp %f, "
                       "local_result %f\n",
                       cta.thread_rank(), offset, static_cast<float>(original),
                       static_cast<float>(local_temp),
                       static_cast<float>(local_result));
            }
        }
    }

    // Write result for this block to global mem.
    if (cta.thread_rank() == 0) {
        g_odata[blockIdx.x] = local_result;
    }
}

template <typename scalar_t, typename func_t>
void LaunchReductionKernelOneOutput(const Indexer& indexer,
                                    scalar_t identity,
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

    ReduceKernelInit<scalar_t><<<grid_size, block_size,
                                 GetSMSize<scalar_t>(grid_size, block_size)>>>(
            indexer, identity, element_kernel, d_odata, n);

    OPEN3D_CUDA_CHECK(cudaMemcpy(indexer.GetOutputPtr(0), d_odata,
                                 sizeof(scalar_t), cudaMemcpyDeviceToHost));
}

}  // namespace cuda_launcher
}  // namespace kernel
}  // namespace open3d
